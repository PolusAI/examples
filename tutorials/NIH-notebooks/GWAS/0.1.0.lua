help([[
    Conda environment with GWAS packages
    ]])

whatis("Version: 0.1.0")
whatis("Keywords: GWAS")

prepend_path("JUPYTER_PATH", "/opt/modules/shared/conda-envs/gwas/share/jupyter")
setenv("R", "/opt/modules/shared/conda-envs/gwas/bin/R")
setenv("RETICULATE_PYTHON", "/opt/modules/shared/conda-envs/gwas/bin/python")
append_path("PATH","/opt/modules/shared/conda-envs/gwas/bin")

