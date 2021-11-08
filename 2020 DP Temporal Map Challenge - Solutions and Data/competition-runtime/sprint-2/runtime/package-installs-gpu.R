# ============ CONFIGURATION ============
# Add conda executables to PATH
Sys.setenv(PATH=paste0("/opt/conda/envs/r-gpu/bin:", Sys.getenv("PATH")))

# Set CRAN repo
r = getOption("repos")
r["CRAN"] = "https://cloud.r-project.org/"
options(repos = r)

install.packages("remotes", quiet = T)
library(remotes)
