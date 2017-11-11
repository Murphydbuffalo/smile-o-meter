# The Smile-O-Meter
A machine vision application that animates a CSS smiley face based on the
current facial expression of the user.

The user's webcam is used to capture a continuous stream of images, which are
fed to a model for detecting happy facial expressions.

The plan right now is to train a model using Python and Numpy, and to then input
the learned paramters into an asm.js script that will reside on the client and
be used to classify the images as containing a happy expression, or not.

More to come!

## Development
Get yourself homebrew and npm, then:
`brew install md5sum`
`npm install -g uglify-es`
`npm install -g uglifycss`

Run `./bin/build.sh` to build the assets.
Run `./bin/deploy.sh` to deploy to AWS assuming you've set up your AWS CLI credentials
and have a static site hosted on S3 and Cloudfront (you'll need to update `deploy.sh`
with your app name).
