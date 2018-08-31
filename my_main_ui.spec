# -*- mode: python -*-

block_cipher = None

import sys
sys.setrecursionlimit(500000) # or more


def get_pandas_path():
    import pandas
    pandas_path = pandas.__path__[0]
    return pandas_path



a = Analysis(['my_main_ui.py'],
             pathex=['C:\\Users\\Administrator.USER-20170417DX\\Desktop\\ML+Pyqt\\code\\my_app'],
             binaries=[('D:/Anaconda3/Lib/site-packages/pyzbar/libiconv-2.dll','.'),('D:/Anaconda3/Lib/site-packages/pyzbar/libzbar-32.dll','.')],
             datas=[],
             hiddenimports=['sklearn','sklearn.neighbors.typedefs','scipy._lib.messagestream','sklearn.neighbors.quad_tree','sklearn.tree._utils'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
			 
dict_tree = Tree(get_pandas_path(), prefix='pandas', excludes=["*.pyc"])
a.datas += dict_tree
a.binaries = filter(lambda x: 'pandas' not in x[0], a.binaries)

exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='my_main_ui',
          debug=False,
          strip=False,
          upx=True,
          console=False,
		  icon='C:\\Users\\Administrator.USER-20170417DX\\Desktop\\ML+Pyqt\\code\\my_app\\icon.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='my_main_ui')
