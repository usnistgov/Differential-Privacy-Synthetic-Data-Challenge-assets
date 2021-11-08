#!/bin/bash

exit_code=0

{
    cd /codeexecution

    # Check for gpu with nvidia-smi
    if [ $(which nvidia-smi) ]
    then
	processor="gpu"
    else
        processor="cpu"
    fi
    echo "Running $processor image"

    echo "Unpacking submission..."
    unzip ./submission/submission.zip -d ./

    if [ -f "main.py" ]
    then
        echo "Running submission with Python"
        conda run -n py-$processor python main.py
    elif [ -f "main.R" ]
    then
        echo "Running submission with R"
        conda run -n r-$processor Rscript main.R
    elif [ -f "main" ]
    then
	if [ $(stat -c %A main | cut -c4) = "x" ]
	then
            echo "Running submission binary"
            ./main
	else
	    echo -e "ERROR: main is not executable. Please run:\n\n\tchmod u+x main\n\nbefore creating your submission."
	    exit_code==1
	fi

    else
        echo "ERROR: Could not find main.py, main.R, or executable main in submission.zip"
        exit_code=1
    fi

    echo "Exporting submission.csv result..."

    # Valid scripts must create a "submission.csv" file within the same directory as main
    if [ -f "submission.csv" ]
    then
        echo "Script completed its run."
        cp submission.csv ./submission/submission.csv
    else
        echo "ERROR: Script did not produce a submission.csv file in the main directory."
        exit_code=1
    fi

    # Test that submission is valid
    conda run -n py-$processor pytest -v

    # Score the submission
    conda run -n py-$processor python scripts/score.py

    echo "================ END ================"
} |& tee "/codeexecution/submission/log.txt"

# copy for additional log uses
cp /codeexecution/submission/log.txt /tmp/log
exit $exit_code
