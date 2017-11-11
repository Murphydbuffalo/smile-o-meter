#!/bin/sh

# Exit the script if any command within it returns with an error.
set -e

./bin/build.sh

# Set `Content-Encoding` to `gzip` for HTML assets.
aws s3 sync ./build s3://smile-o-meter.rocks --acl public-read --exclude "*" --include "*.html"  --content-encoding "gzip"

# Set `Content-Encoding` to `gzip` and set `Cache-Control` to `maxage=31536000` for CSS & JS manifest files.
aws s3 sync ./build s3://smile-o-meter.rocks --acl public-read --exclude "*" --include "*.css" --include "*.js" --content-encoding "gzip" --cache-control "max-age=31536000"

# Set `Cache-Control` to `maxage=31536000` for images.
aws s3 sync ./build s3://smile-o-meter.rocks --acl public-read --exclude "*" --include "images/*" --cache-control "max-age=31536000"

# Upload remaining files (favicons) as-is.
aws s3 sync ./build s3://smile-o-meter.rocks --acl public-read --exclude "*.html" --exclude "*.css" --exclude "*.js" --exclude "images/*"
