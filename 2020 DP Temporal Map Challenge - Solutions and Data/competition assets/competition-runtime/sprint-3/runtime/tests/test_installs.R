# This script tests if packages can be loaded

library(futile.logger)
futile.logger::flog.threshold(futile.logger::INFO)

flog.info("Testing if R packages can be loaded correctly.")

packages_essentials <- c(
    "ggplot2",
    "plyr",
    "reshape2",
    "dplyr",
    "tidyr",
    "caret",
    "randomForest",
    "data.table",
    "quantmod",
    "shiny",
    "rmarkdown",
    "glmnet",
    "jsonlite",
    "zoo",
    "rbokeh",
    "formatR",
    "tidyverse",
    "DBI",
    "broom",
    "forcats",
    "haven",
    "hms",
    "httr",
    "lubridate",
    "magrittr",
    "modelr",
    "purrr",
    "readr",
    "readxl",
    "rvest",
    "stringr",
    "tibble",
    "xml2"
)

packages <- c(
    packages_essentials,
    "reticulate",
    "keras",
    "tensorflow",
    ## ADD ADDITIONAL REQUIREMENTS BELOW HERE ##

    ############################################
    NULL
)

for (package in packages) {
    flog.info(sprintf("Testing if %s can be loaded...", package))
    library(package, character.only = TRUE)
}

flog.info("All required packages successfully loaded.")
