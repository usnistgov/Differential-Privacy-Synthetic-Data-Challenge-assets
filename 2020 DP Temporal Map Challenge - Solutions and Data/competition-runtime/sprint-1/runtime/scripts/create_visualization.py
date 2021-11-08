import json
from pathlib import Path

import jinja2
import requests
from loguru import logger
import typer

JINJA_TEMPLATE_URL = "https://drivendata-competition-deid2-public.s3.amazonaws.com/visualization/report.jinja2"


def main(json_report: Path, params_file: Path, template_path: Path = None):
    """
    Take the output of a `score.py` run and create an interactive HTML/JS visualization.
    """
    for expected_file in (json_report, params_file):
        if not expected_file.exists():
            raise typer.Exit(f"file not found: {expected_file}")

    logger.info(f"reading report from {json_report} ...")
    json_report = json.loads(json_report.read_text())

    logger.info(f"reading parameters from {params_file} ...")
    params_file = json.loads(params_file.read_text())
    # remove unnecessary clutter we don't use in the visualization
    params_file["schema"].pop("incident_type")

    if template_path is None:
        logger.info(f"downloading template from {JINJA_TEMPLATE_URL}...")
        r = requests.get(JINJA_TEMPLATE_URL)
        if not r.status_code == 200:
            logger.error(f"could not download template! error {r.status_code}")
        template_text = r.content.decode("utf8")
    else:
        template_text = template_path.read_text()

    context = {"report": json.dumps(json_report), "parameters": json.dumps(params_file)}

    logger.info("rendering html...")
    env = jinja2.Environment()
    template = env.from_string(template_text)
    html = template.render(**context)
    typer.echo(html)


if __name__ == "__main__":
    typer.run(main)
