# -*- mode: python ; coding: utf-8 -*-
import os

# Force PySide6 bindings for pyqtgraph and matplotlib Qt backend
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'
os.environ['QT_API'] = 'PySide6'


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'sphinx', 'pytest', 'py', 'pygments', 'docutils',
        'setuptools', 'pkg_resources', 'importlib_metadata', 'importlib_resources',
        'PyQt6', 'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='svm-analyst',  # v1.1.4
    icon='assets/svm-analyst.ico',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
