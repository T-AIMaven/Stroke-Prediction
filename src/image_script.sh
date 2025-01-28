%%bash -s "$LOCAL_MODE" "$IMAGE_NAME"
# | eval: false

algorithm_name=$2
account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration
# (default to us-east-1 if none defined)
region=$(aws configure get region)
region=${region:-eu-central-1}

repository="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

# We only want to push the Docker image to ECR if
# we are not running in Local Mode.
if [ $1 = "False" ]
then
    # Create the repository if it doesn't exist in ECR
    aws ecr describe-repositories \
        --repository-names "${algorithm_name}" > /dev/null 2>&1
    if [ $? -ne 0 ]
    then
        aws ecr create-repository \
            --repository-name "${algorithm_name}" > /dev/null
    fi

    # Get the login command from ECR to run the
    # Docker push command.
    aws ecr get-login-password \
        --region ${region}|docker \
        login --username AWS --password-stdin ${repository}

    # Push the Docker image to the ECR repository
    docker tag ${algorithm_name} ${repository}
    docker push ${repository}
fi