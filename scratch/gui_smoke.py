import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from PySide6.QtWidgets import QApplication
from surface_fairing.gui.main_window import FairingWindow
app=QApplication([])
win=FairingWindow(); win.testing=True
for idx in range(3):
    win.artifact_combo.setCurrentIndex(idx)
    win._run_fairing()
    r=win.result
    assert r is not None and r.hard_constraint_error < 1e-8
    txt=win.message.toPlainText()
    for token in ("弯曲能量模式","高频波纹 RMS","硬约束最大误差","平均曲率 RMS"):
        assert token in txt, (idx,token)
    print("artifact",idx,"mode",r.energy_model.bend_mode,
          "hard %.1e"%r.hard_constraint_error,"plots OK; msg lines",len(txt.splitlines()))
# exercise each bend mode on curved artifact
win.artifact_combo.setCurrentIndex(1)
for mi in range(3):
    win.bend_mode.setCurrentIndex(mi)
    win._run_fairing()
    print("curved bend-mode",win.bend_mode.currentText().split()[0],"ok")
# error path: uncheck all edges -> must surface exception, not crash
for chk in win.edge_locks.values(): chk.setChecked(False)
win._run_fairing()
assert "错误" in win.message.toPlainText()
print("error surfacing OK")
# maintenance path: conflicting offset key point -> analyze -> prune & recompute
for chk in win.edge_locks.values(): chk.setChecked(True)
win.key_offset.setChecked(True)
win._analyze_constraints()
assert "最小不可满足" in win.message.toPlainText()
assert win.remove_combo.count() >= 2
win._prune_and_recompute()
assert win.result is not None and win.result.hard_constraint_error < 1e-8
assert "重新生效条件" in win.message.toPlainText()
print("maintenance prune path OK")
# relaxation boundary path
win.key_offset.setChecked(False); win.soft_lock.setChecked(True)
win._analyze_constraints()
assert "松弛边界" in win.message.toPlainText()
print("maintenance relax path OK")
print("SMOKE PASS")
