@echo off
rem Builds a single Windows program of the logger. Run from the repository folder.
pip install pyinstaller
pyinstaller --noconfirm --onefile --windowed --name km-logger ^
  --add-data "configs;configs" --collect-submodules km --collect-submodules logger ^
  logger\run.py
echo.
echo The program is in dist\km-logger.exe
