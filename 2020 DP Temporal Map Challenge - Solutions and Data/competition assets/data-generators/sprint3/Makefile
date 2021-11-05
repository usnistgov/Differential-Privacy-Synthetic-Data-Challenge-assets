.PHONY: update update-prescreen update-platform
DEFAULT_GOAL := update

update: update-prescreen update-platform

update-platform:
	aws s3 cp ../../data/sprint3/final/public/ground_truth.csv s3://drivendata-prod/data/76/private/ground_truth.csv
	aws s3 cp ../../data/sprint3/final/public/ground_truth.csv s3://drivendata-prod/data/76/public/ground_truth.csv
	aws s3 cp ../../data/sprint3/final/prescreened/ground_truth.csv s3://drivendata-prod/data/77/private/ground_truth.csv

update-prescreen:
	az storage blob upload --account-name deid2storageprod2 --container-name data --name parameters.json --file ../../../deid2-runtime/data/parameters.json
	az storage blob upload --account-name deid2storageprod2 --container-name data --name ground_truth.csv --file ../../data/sprint3/final/prescreened/ground_truth.csv
