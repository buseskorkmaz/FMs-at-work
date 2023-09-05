#!/bin/bash

for i in {1..99}
do
    jbsub -queue x86_1h -mem 48g -cores 4+1 python generate_user_profile_embeddings.py --index $i
done
