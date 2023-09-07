#!/bin/bash

for i in {0..199}
do
    jbsub -queue x86_12h -mem 32g -cores 4+0 python q_values_of_job_descriptions.py --index $i
done
