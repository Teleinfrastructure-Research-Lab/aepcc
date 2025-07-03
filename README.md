# Install

```bash
git clone --recurse-submodules git@github.com:IvayloBozhilov/ae-pcc.git
python install_dependencies.py
pip install -e .
pip install submodules/smol
mv config/out_of_project_paths.yaml.example config/out_of_project_paths.yaml
# enter the correct paths
mkdir runs
mkdir runs/foldingnet
mkdir runs/gae
mkdir runs/tgae
```
