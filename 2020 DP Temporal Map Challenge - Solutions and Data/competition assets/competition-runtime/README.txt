The Algorithm Contest of the Differential Privacy Temporal Map Challenge (DeID2)
ran on DrivenData from October 2020 to June 2021. The contest included 3 sprints, 
each with its own data set, scoring metric, and leaderboard:


Sprint 1: Baltimore 911 Incidents
* [Problem Description](https://www.drivendata.org/competitions/69/deid2-sprint-1-prescreened/page/263/)
* [Results](https://www.drivendata.co/blog/differential-privacy-winners-sprint1)


Sprint 2: American Community Survey
* [Problem Description](https://www.drivendata.org/competitions/75/deid2-sprint-2-prescreened/page/285/)
* [Results](https://www.drivendata.co/blog/differential-privacy-winners-sprint2)


Sprint 3: Chicago Taxi Rides
* [Problem Description](https://www.drivendata.org/competitions/77/deid2-sprint-3-prescreened/page/332/)
* [Results](https://www.drivendata.co/blog/differential-privacy-winners-sprint3)


The folders included here contain the data and scoring assets from each sprint.
These are snapshots of the runtime repository used for final scoring. In each
folder, the `data/` directory contains the default data that was used for the
public leaderboard. But there are alternate directories for the final scoring
data sets -- e.g. in sprint 2 the `data-GA-NC-SC` and `data-NY-PA` folders --
which contain the data sets used for final scoring.


In order to do a final scoring run, just rename one of these final scoring data
folders to `data` and then follow the directions in the folder's `README` to
run with `docker`. Here's the pseudocode for getting all the final scores for
sprint 3 for example:


    for submission_dir in submissions:
        for dataset in ('data2016', 'data2020'):
            for i in # of desired repetitions per dataset:
                rename dataset/ to data/
                copy submission_dir/submission.zip into submission/
                run the runtime container
                collect the results


Here's an actual shell script used to run a single time, mapping the output
directory to a temporary folder:


        #!/usr/bin/env bash
        set -euxo pipefail


        # usage
        [ $# -ne 2 ] && { echo "Usage: $0 <path/to/submission.zip> <data, e.g. '2016'>"; exit 1; }
        # constants
        SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
        PROJECT_ROOT="/tmp/deid2-collaboration"


        # name the args
        submission_path=$1
        submission_dir=$(dirname $1)
        submission_uuid=$(basename $submission_dir)
        data_set=$2
        data_path="${PROJECT_ROOT}/sprint3-final-scoring/data${data_set}"
        output_path="${PROJECT_ROOT}/sprint3-final-scoring/results/${submission_uuid}/${data_set}"


        echo "submission_path: ${submission_path}"
        echo "submission_uuid: ${submission_uuid}"
        echo "data_set: ${data_set}"
        echo "data_path: ${data_path}"
        echo "output_path: ${output_path}"


        # checks
        [ ! -f $submission_path ] && { echo "File not found: ${submission_path}"; exit 1; }
        [ ! -f "${data_path}/ground_truth.csv" ] && { echo "File not found: ${data_path}/ground_truth.csv"; exit 1; }


        # set up the submission
        tmp_dir=$(mktemp -d)
        chmod 777 $tmp_dir
        cp $submission_path $tmp_dir
        echo "created temp dir: ${tmp_dir}"


        # note when we start
        date -Iseconds > ${tmp_dir}/start


        # run docker
        echo "running docker..."
        docker run -i \
            --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
            --gpus all \
            --network none \
            --shm-size 8g \
            --mount type=bind,source="${data_path}",target=/codeexecution/data,readonly \
            --mount type=bind,source="${tmp_dir}",target=/codeexecution/submission \
            drivendata/deid2-competition:gpu-latest


        # note when we end
        date -Iseconds > ${tmp_dir}/end


        # copy things back to where they should go
        mkdir -p $output_path
        mv $tmp_dir $output_path/


This results in a directory with the resulting `submission.csv` but also the `log.txt`
as well as files containing the start and stop times (for tracking length of run).