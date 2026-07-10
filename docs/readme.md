# Readme

- python setup in project directory using venv (`/Users/darbyjack/miniforge3/bin/python3 -m venv .venv`)
- `pip install neurokit2 numpy scipy pandas pyarrow pytest`


to activate python 

cd /Users/darbyjack/hrv-pipeline 
source .venv/bin/activate

data leak check: `git ls-files | grep -i csv      # must return nothing`

https://bitbucket.org/movesense/movesense-device-lib/downloads/