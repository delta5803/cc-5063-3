"""Surface fairing editing window (primary interface).

The window lets the user pick a parametric test surface, declare boundary /
key-point / soft-ring constraints and metric scales, run the actual energy
fairing, and inspect computed curvature-change fields, position residuals and
reflection stripes (isophotes) before and after. All plots and numbers come
from real computations; errors are surfaced as dialog messages rather than
silently hidden. No normal smoothing or control-point averaging is used for
the displayed results.
"""
from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QTextEdit,
    QVBoxLayout, QWidget)

from ..core.bspline import Metric
from ..core.constraints import HardConstraints, make_ring_soft_lock
from ..core.curvature import curvature_change, curvature_field, isophote_field
from ..core.fairing import fair_surface
from ..io_.artifacts import (
    over_fairing_case, rippled_curved, rippled_plane)
from ..io_.export import export_obj, write_report


class FairingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("曲面光顺编辑视窗 / Surface Fairing Editor")
        self.resize(1480, 920)
        self.result = None
        self._build_widgets()
        self._load_surface()

    # ------------------------------------------------------------------ layout
    def _build_widgets(self):
        root = QHBoxLayout(self)

        control = QVBoxLayout()
        root.addLayout(control, 0)

        art_box = QGroupBox("1. 测试曲面 / Test surface")
        ab = QVBoxLayout(art_box)
        self.artifact_combo = QComboBox()
        self.artifact_combo.addItems(
            ["rippled_plane 带扰动平面",
             "rippled_curved 带扰动弯曲面",
             "feature_bump 需保留特征(过度光顺案例)"])
        self.artifact_combo.currentIndexChanged.connect(self._load_surface)
        ab.addWidget(self.artifact_combo)
        control.addWidget(art_box)

        metric_box = QGroupBox("2. 声明度量 / Declared metric")
        mb = QVBoxLayout(metric_box)
        self.metric_u = self._spin(mb, "scale_u (物理尺度)", 1.0, 0.05, 10.0)
        self.metric_v = self._spin(mb, "scale_v (物理尺度)", 1.0, 0.05, 10.0)
        self.metric_note = QLabel(
            "限制: 采用常系数尺度, 不能表达非均匀参数化; 二次型冻结参考面"
            "积元/法向, 对弯曲面是小位移线性近似, 报告中的直接积分值才是"
            "光顺后真实几何。")
        self.metric_note.setWordWrap(True)
        mb.addWidget(self.metric_note)
        control.addWidget(metric_box)

        con_box = QGroupBox("3. 约束 / Constraints")
        cb = QVBoxLayout(con_box)
        self.edge_locks = {}
        for edge in ("u0", "u1", "v0", "v1"):
            chk = QCheckBox(f"固定边界 {edge} (硬约束)")
            chk.setChecked(True)
            cb.addWidget(chk)
            self.edge_locks[edge] = chk
        self.key_lock = QCheckBox("保留中心点 S(0.5,0.5) (关键点硬约束)")
        cb.addWidget(self.key_lock)
        self.soft_lock = QCheckBox("u0 邻环软锁定 (冲突时可松弛)")
        cb.addWidget(self.soft_lock)
        self.soft_weight = self._spin(cb, "软约束权重", 100.0, 0.0, 1e5,
                                      decimals=2)
        control.addWidget(con_box)

        opt_box = QGroupBox("4. 能量权重 / Energy weights")
        ob = QVBoxLayout(opt_box)
        self.bend_mode = QComboBox()
        self.bend_mode.addItems(
            ["jerk 曲率变化(默认, 保留光滑基底)",
             "normal 法向弯曲(小位移, 弯曲面易压平)",
             "thin_plate 薄板二阶(对比基准)"])
        ob.addWidget(QLabel("弯曲能量类型 / bend mode"))
        ob.addWidget(self.bend_mode)
        self.bend_w = self._spin(ob, "弯曲能量权重", 1.0, 0.0, 100.0)
        self.memb_w = self._spin(ob, "膜能量权重", 0.0, 0.0, 100.0)
        self.attach_w = self._spin(ob, "数据附着权重(防漂移/控形)", 0.2,
                                   0.001, 100.0, decimals=3)
        self.gauss_order = QSpinBox()
        self.gauss_order.setRange(2, 4)
        self.gauss_order.setValue(3)
        ob.addWidget(QLabel("每节点跨 Gauss 积分阶数"))
        ob.addWidget(self.gauss_order)
        self.energy_note = QLabel(
            "能量矩阵已无量纲归一化, 附着权重在所有模式下含义一致; "
            "直接积分能量仍按物理量纲报告。")
        self.energy_note.setWordWrap(True)
        ob.addWidget(self.energy_note)
        control.addWidget(opt_box)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("运行光顺 / Fair")
        self.run_btn.clicked.connect(self._run_fairing)
        self.export_btn = QPushButton("导出 OBJ + 报告")
        self.export_btn.clicked.connect(self._export)
        self.export_btn.setEnabled(False)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.export_btn)
        control.addLayout(run_row)

        self.message = QTextEdit()
        self.message.setReadOnly(True)
        self.message.setMinimumWidth(360)
        control.addWidget(self.message, 1)

        # figure area
        fig_holder = QVBoxLayout()
        root.addLayout(fig_holder, 1)
        self.fig = Figure(figsize=(10, 9))
        self.canvas = FigureCanvas(self.fig)
        fig_holder.addWidget(NavToolbar(self.canvas, self))
        fig_holder.addWidget(self.canvas)

    def _spin(self, layout, label, value, lo, hi, decimals=4):
        layout.addWidget(QLabel(label))
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setValue(value)
        layout.addWidget(sp)
        return sp

    # --------------------------------------------------------------- data flow
    def _load_surface(self):
        idx = self.artifact_combo.currentIndex()
        if idx == 0:
            self.artifact = rippled_plane()
            self.bend_mode.setCurrentIndex(1)  # normal: removes planar ripple
            self.attach_w.setValue(0.3)
        elif idx == 1:
            self.artifact = rippled_curved()
            self.bend_mode.setCurrentIndex(0)  # jerk: keeps the smooth dome
            self.attach_w.setValue(0.2)
        else:
            self.artifact = over_fairing_case()
            self.bend_mode.setCurrentIndex(0)  # jerk: feature retention
            self.attach_w.setValue(0.2)
        self.surface = self.artifact.surface
        self.result = None
        self.export_btn.setEnabled(False)
        self._message(f"已加载 {self.artifact.name}: {self.artifact.description}")
        self._plot_initial()

    def _build_constraints(self):
        surf = self.surface
        surf.metric = Metric(self.metric_u.value(), self.metric_v.value())
        hard = HardConstraints(3 * surf.n_control)
        locked = [e for e, chk in self.edge_locks.items() if chk.isChecked()]
        if not locked:
            raise ValueError("至少需要固定一条边界, 否则约束不足会导致整体漂移")
        for edge in locked:
            hard.pin_edge(surf, edge)
        if self.key_lock.isChecked():
            hard.pin_surface_point(surf, 0.5, 0.5)
        soft = ([make_ring_soft_lock(surf, "u0", self.soft_weight.value())]
                if self.soft_lock.isChecked() else [])
        return hard, soft, locked

    def _run_fairing(self):
        try:
            hard, soft, locked = self._build_constraints()
            bend_mode = self.bend_mode.currentText().split()[0]
            self.result = fair_surface(
                self.surface, hard, soft=soft,
                gauss_order=self.gauss_order.value(),
                bend_weight=self.bend_w.value(),
                bend_mode=bend_mode,
                memb_weight=self.memb_w.value(),
                attachment_weight=self.attach_w.value(),
                tangent_edges=tuple(locked))
            self.export_btn.setEnabled(True)
            self._report_numbers()
            self._plot_results()
        except Exception as exc:  # surface failures must be surfaced, not hidden
            detail = traceback.format_exc()
            self._message("错误 / ERROR: " + str(exc))
            if not getattr(self, "testing", False):
                QMessageBox.critical(
                    self, "光顺失败 / Fairing error",
                    f"{exc}\n\n{detail}")

    # ------------------------------------------------------------------ plots
    def _plot_initial(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection="3d")
        _, _, pts = self.surface.sample_grid(28, 28)
        ax.plot_surface(pts[:, :, 0], pts[:, :, 1], pts[:, :, 2],
                        cmap="viridis", alpha=0.85)
        ax.set_title(f"Input surface: {self.surface.name}")
        self.canvas.draw()

    def _plot_results(self):
        r = self.result
        self.fig.clear()
        before, after = r.original, r.faired
        grid = 24

        ax1 = self.fig.add_subplot(2, 3, 1, projection="3d")
        self._surface(ax1, before, "Before fairing")
        ax2 = self.fig.add_subplot(2, 3, 2, projection="3d")
        self._surface(ax2, after, "After fairing")

        change = curvature_change(before, after, grid, grid)
        ax3 = self.fig.add_subplot(2, 3, 3)
        im = ax3.imshow(change["mean_abs_delta"], origin="lower", cmap="magma")
        ax3.set_title("Mean curvature change |dH| (computed)")
        self.fig.colorbar(im, ax=ax3, fraction=0.046)

        # position residual field, evaluated pointwise (not control averaging)
        us = np.linspace(0, 1, grid)
        vs = np.linspace(0, 1, grid)
        resid = np.zeros((grid, grid))
        for i, u in enumerate(us):
            for j, v in enumerate(vs):
                resid[i, j] = np.linalg.norm(
                    after.evaluate(u, v) - before.evaluate(u, v))
        ax4 = self.fig.add_subplot(2, 3, 4)
        im2 = ax4.imshow(resid, origin="lower", cmap="cividis")
        ax4.set_title("Position residual (sampled distance)")
        self.fig.colorbar(im2, ax=ax4, fraction=0.046)

        light = np.array([0.3, 0.4, 1.0])
        _, _, iso_before = isophote_field(before, light, grid, grid)
        _, _, iso_after = isophote_field(after, light, grid, grid)
        ax5 = self.fig.add_subplot(2, 3, 5)
        c5 = ax5.contour(iso_before, levels=10, cmap="Reds")
        ax5.clabel(c5, inline=True, fontsize=6)
        ax5.set_title("Reflection stripes before (isophotes)")
        ax6 = self.fig.add_subplot(2, 3, 6)
        c6 = ax6.contour(iso_after, levels=10, cmap="Blues")
        ax6.clabel(c6, inline=True, fontsize=6)
        ax6.set_title("Reflection stripes after (isophotes)")
        self.fig.tight_layout()
        self.canvas.draw()

    @staticmethod
    def _surface(ax, surf, title):
        _, _, pts = surf.sample_grid(28, 28)
        ax.plot_surface(pts[:, :, 0], pts[:, :, 1], pts[:, :, 2],
                        cmap="viridis", alpha=0.85)
        ax.set_title(title)

    # --------------------------------------------------------------- reporting
    def _report_numbers(self):
        r = self.result
        _, _, cb = curvature_field(r.original)
        _, _, ca = curvature_field(r.faired)
        hf0 = self._high_frequency_rms(r.original)
        hf1 = self._high_frequency_rms(r.faired)
        lines = [
            f"弯曲能量模式: {r.energy_model.bend_mode}",
            f"度量: {r.metric_label}",
            f"Gauss 积分点(逐节点跨): {r.energy_model.n_integration_points}",
            f"能量归一化因子: {r.energy_model.energy_scale:.4e}",
            f"能量(直接积分) {r.direct_before:.6f} -> {r.direct_after:.6f}",
            f"能量下降比例: {r.energy_reduction_ratio:.2%}",
            f"高频波纹 RMS: {hf0:.6f} -> {hf1:.6f}",
            f"硬约束最大误差: {r.hard_constraint_error:.3e}",
            f"最大位置残差: {r.position_residual:.6f}",
            f"平均曲率 |H| 峰值: {np.abs(cb['mean']).max():.4f} -> "
            f"{np.abs(ca['mean']).max():.4f}",
            f"平均曲率 RMS: {np.sqrt(np.mean(cb['mean']**2)):.4f} -> "
            f"{np.sqrt(np.mean(ca['mean']**2)):.4f}",
            f"高斯曲率 |K| 峰值: {np.abs(cb['gaussian']).max():.4f} -> "
            f"{np.abs(ca['gaussian']).max():.4f}",
            f"求解状态: {r.report.message}; ridge={r.report.ridge_used:.2e}; "
            f"cond={r.report.condition_number:.2e}",
        ]
        for edge, tr in r.tangential_ranges.items():
            lo, hi = tr.angle_envelope_deg()
            lines.append(f"边 {edge} 可达切向角范围: [{lo:.4f}°, {hi:.4f}°]")
        if r.soft_events:
            for ev in r.soft_events:
                lines.append(
                    f"软约束松弛: {ev['description']} 环={ev['ring_indices']} "
                    f"最大偏离={ev['max_deviation']:.6f}")
        else:
            lines.append("无软约束松弛事件。")
        self._message("\n".join(lines))

    @staticmethod
    def _high_frequency_rms(surf, grid: int = 48, cutoff: int = 3) -> float:
        """RMS of the spectral high-frequency part of the sampled z-field."""
        ts = np.linspace(0.0, 1.0, grid)
        Z = np.array([[surf.evaluate(u, v)[2] for v in ts] for u in ts])
        F = np.fft.fft2(Z - Z.mean())
        k = np.minimum(np.arange(grid), grid - np.arange(grid))
        KU, KV = np.meshgrid(k, k, indexing="ij")
        mask = (np.abs(KU) >= cutoff) | (np.abs(KV) >= cutoff)
        return float(np.sqrt(np.mean(np.real(np.fft.ifft2(F * mask)) ** 2)))

    def _export(self):
        if self.result is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录", "output")
        if not folder:
            return
        folder = Path(folder)
        obj = export_obj(self.result.faired,
                         folder / f"{self.surface.name}_faired.obj")
        payload = {
            "surface": self.surface.name,
            "metric": self.result.metric_label,
            "energy_before": self.result.direct_before,
            "energy_after": self.result.direct_after,
            "reduction_ratio": self.result.energy_reduction_ratio,
            "hard_constraint_error": self.result.hard_constraint_error,
            "position_residual": self.result.position_residual,
            "soft_events": self.result.soft_events,
        }
        report = write_report(payload, folder / f"{self.surface.name}_report.json")
        self._message(f"已导出真实曲面: {obj}\n报告: {report}")

    def _message(self, text: str):
        self.message.append(text)


def main() -> int:
    from PySide6.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    win = FairingWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
