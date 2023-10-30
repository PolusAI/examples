help([[
    Conda environment with Pangolin packages
    ]])

whatis("Version: 0.1.0")
whatis("Keywords: Pangolin")

prepend_path("JUPYTER_PATH", "/opt/modules/shared/conda-envs/pangolin/share/jupyter")
setenv("JUPYTER_KERNEL_NAME", "Pangolin")
setenv("PYTHON_EXEC_PATH", "/opt/modules/shared/conda-envs/pangolin/bin/python")
append_path ("PATH", "/opt/modules/shared/conda-envs/pangolin/bin")
