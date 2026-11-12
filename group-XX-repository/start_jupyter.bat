@echo off
rem Double-click to open Jupyter Notebook in this project folder, using the ml-project environment.
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" ml-project
cd /d "%~dp0"
jupyter notebook
