# offline-discharge-data-product-creation
Offline Discharge Data Product Creation

***Example Usage***

```bash
InputDir=Path('./offline_inputs')
reaches_json=InputDir.joinpath('reaches.json')

with open(reaches_json) as json_file:
    reaches = json.load(json_file)
    
nR=len(reaches)

for reach in range(nR):
     %run /Users/mtd/GitHub/SWOT-confluence/offline-discharge-data-product-creation/run_offline.py 'unconstrained' 'timeseries' 'reaches.json'  $reach
     
```

So the command line arguments in order are [branch] [SWOT observation mode] [reaches file] [reach #]

where [branch] is either "constrained" or "unconstrained"
and [SWOT observation mode] is either "timeseries" or "single_pass". The "timeseries" option uses the .nc files created by the input module, and the "single_pass" uses the L2 single pass SWOT shapefiles.

## deployment

There is a script to deploy the Docker container image and Terraform AWS infrastructure found in the `deploy` directory.

Script to deploy Terraform and Docker image AWS infrastructure

REQUIRES:

- jq (<https://jqlang.github.io/jq/>)
- docker (<https://docs.docker.com/desktop/>) > version Docker 1.5
- AWS CLI (<https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>)
- Terraform (<https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli>)

Command line arguments:

[1] registry: Registry URI
[2] repository: Name of repository to create
[3] prefix: Prefix to use for AWS resources associated with environment deploying to
[4] s3_state_bucket: Name of the S3 bucket to store Terraform state in (no need for s3:// prefix)
[5] profile: Name of profile used to authenticate AWS CLI commands

Example usage: ``./deploy.sh "account-id.dkr.ecr.region.amazonaws.com" "container-image-name" "prefix-for-environment" "s3-state-bucket-name" "confluence-named-profile"`

Note: Run the script from the deploy directory.
