# Useful scripts for local testing

We provide a number of scripts that you are free to use for your own local development and
testing.

**You aren't required to use any of these scripts** but they carry out some routine
tasks, so here are a few examples of what you might like to use them for:

- Validating a submission you create with your own code.
- Scoring a submission you create with your own code using the custom metric.
- Creating a visualization to explore how well your submission did on various
  neighborhoods over time.

Some of these scripts by default look for data in the locations expected to be mounted within
the Docker container, but we give example invocations of each that you can use locally assuming
you have installed the requirements expected within the container
(see [runtime/py-cpu.yml](runtime/py-cpu.yml) or the simplified [requirements.txt](requirements.txt)
in this dir with a Python 3.8 environment).

(If you see us refer to `/tmp` in any of the invocations below, that means we expect you to
have used one of these scripts to create that file.)

---

## 1. `create_submission_format.py`

This script will create a properly formatted submission, albeit with all zeros for the privatized
counts. It does so using only the `parameters.json` file, meaning that for the purposes of
differential privacy it has not "looked" at the ground truth data at all.

### 1.1. Usage

```
Usage: create_submission_format.py [OPTIONS] PARAMS_FILE

  Create a barebones submission file with all zeros for counts.

  If `--output-file` is not provided then the script will print output to
  stdout.

Arguments:
  PARAMS_FILE  [required]

Options:
  --output-file PATH

  --help                          Show this message and exit.
```

### 1.2. Example for local use

```
python runtime/scripts/create_submission_format.py data/parameters.json \
   --output-file /tmp/submission_format.csv
```

---

## 2. `score.py` and `metric.py`

This `metric.py` file contains the implementation of the scoring metric so that you can
use it locally, %read it into a Jupyter notebook, import it in a Python script, or reimplement
it in other languages. See the problem description for an explanation of the metric.

The `score.py` script takes a ground truth file (see above) and a privatized submission
and calculates a score. It can also optionally output a JSON report to help you debug which
neighborhood-year-month permutations you are having the most trouble with. This JSON report
is also used by the `create_visualization.py` script.

### 2.1. Usage

```
Usage: score.py [OPTIONS]

  Given a raw incidents file and a submission file, calculate the score that
  the submission would receive if submitted.

Options:
  --incidents-csv PATH            [default: /codeexecution/data/incidents.csv]
  --submission-csv PATH           [default: /codeexecution/submission.csv]
  --json-report / --no-json-report
                                  [default: False]

  --help                          Show this message and exit.
```

### 2.2. Example for local use

Let's see what score we would get from submitting the all-zero submission format we
just created in the steps above:

```
python runtime/scripts/score.py \
  --incidents-csv data/incidents.csv \
  --submission-csv /tmp/submission_format.csv
``` 

If you want to save the JSON report, just add the `--json-report` flag as follows and send
the output to a file so you can use it:

```
python runtime/scripts/score.py \
  --incidents-csv data/incidents.csv \
  --submission-csv /tmp/submission_format.csv \
  --json-report > /tmp/report.json
```

The visualization script uses this report output, but you may wish to use it for exploring
which neighborhood-year-month groups have presented the most difficulty.

## 3. `create_visualization.py`

Given the score report output above, generate a useful visualization which may
be helpful for troubleshooting where the privatization needs attention. If you open
the generated HTML file in your browser and let it load, you should see a choropleth
of Baltimore:

![](https://drivendata-competition-deid2-public.s3.amazonaws.com/visualization/screenshot.png)

### 3.1. Usage

```
Usage: create_visualization.py [OPTIONS] JSON_REPORT PARAMS_FILE

  Take the output of a `score.py` run and create an interactive HTML/JS
  visualization.

Arguments:
  JSON_REPORT  [required]
  PARAMS_FILE  [required]

Options:
  --template-path PATH [only used for testing, or if you'd like to tweak the template]

  --help                          Show this message and exit.
```

### 3.2. Example for local use

We pass in the report created above, the parameters file, and send the output to
a file that we can open locally in the browser.

```
python runtime/scripts/create_visualization.py \
  /tmp/report.json data/parameters.json \
  > /tmp/report.html
```

## 4. `benchmark/main.py`

This file is an example of a simplistic benchmark that you could submit for code execution
which shows the most naive yet still properly privacy-preserving approach: adding Laplace
noise calculated from the specified epsilon for each run.

The main `README.md` for this repo shows how to use this for actual code execution, but you
can use it locally if you'd like.

### 4.1. Usage

```
Usage: main.py [OPTIONS]

  Generate an example submission. The defaults are set so that the script
  will run successfully without being passed any arguments, invoked only as
  `python main.py`.

Options:
  --submission-format PATH        [default:
                                  /codeexecution/data/submission_format.csv]

  --incident-csv PATH             [default: /codeexecution/data/incidents.csv]
  --output-file PATH              [default: /codeexecution/submission.csv]
  --params-file PATH              [default:
                                  /codeexecution/data/parameters.json]

  --random-seed INTEGER           [default: 42]

  --help                          Show this message and exit.
```

### 4.2. Example for local use 

```
python benchmark/main.py \
  --submission-format /tmp/submission_format.csv \
  --incident-csv data/incidents.csv \
  --output-file /tmp/example-submission.csv \
  --params-file data/parameters.json
``` 