#!/bin/sh

# Exit the script if any command within it returns with an error.
set -e

./bin/build.sh

# Sync local build directory with S3 Bucket
aws s3 sync ./build s3://smile-o-meter.rocks --acl public-read --exclude "*.css" --exclude "*.js" --exclude "images/*"
aws s3 sync ./build s3://smile-o-meter.rocks --acl public-read --exclude "*" --include "*.css" --include "*.js" --include "images/*" --cache-control "max-age=31536000"
