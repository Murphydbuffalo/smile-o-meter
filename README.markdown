# The Smile-O-Meter
A machine vision application that animates a CSS smiley face based on the
current facial expression of the user.

The user's webcam is used to capture a continuous stream of images, which are
fed to a model for detecting happy facial expressions.

## Running the app locally
1. Get npm, then `npm install`.
2. Run `./bin/build.sh` to build the assets.
3. `open ./build/index.html`
