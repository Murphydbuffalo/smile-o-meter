#!/bin/sh

# Exit the script if any command within it returns with an error.
set -e

npm install
rm -rf ./build/*

# Webpage assets
cp -R ./favicons/* ./images ./build

# Neural network learned parameters and statistics for input data normalization
cp ./neural_net/output/learned_weights.npy ./neural_net/output/learned_biases.npy ./build
cp ./neural_net/output/training_set_feature_means.npy ./neural_net/output/training_set_zero_mean_feature_standard_deviations.npy ./build

# Concatenate and minify all files in the `/stylesheets` and `/javascript` directories.
# And move the resulting manifest files to the `/build` directory of the project.
#
# We also give each manifest file a unique name that includes the hash of the file
# contents of the corresponding file (either `index.css` and `index.js`).
# These unique names, combined with setting the `Cache-Control` header's `maxage`
# value allow browsers to avoid making requests for our JS and CSS files for up
# to one year, so long as the contents of `index.css/js` haven't changed.
# We'll do this manually for our CSS and use Webpack to do so for our JS so that
# we can use ES6 modules.
CSS_HASH=`./node_modules/md5-file/cli.js ./stylesheets/index.css`
./node_modules/uglifycss/uglifycss --output "./build/index-$CSS_HASH.min.css" ./stylesheets/normalize.css ./stylesheets/skeleton.css ./stylesheets/index.css
./node_modules/webpack-cli/bin/webpack.js

# Build full HTML file from `template.html`
#
# Replace the CSS and JS file names in `template.html` with the hash-based filenames for our CSS and JS manifest files.
# And write the resulting HTML to `template_with_manifests.html`.
JS_FILENAME=`find ./build -name "index-*.min.js" -exec basename {} \;`
CSS_FILENAME=`find ./build -name "index-*.min.css" -exec basename {} \;`
sed "s/index.min.js/$JS_FILENAME/" ./template.html > ./build/index.html
sed -i "" "s/index.min.css/$CSS_FILENAME/" ./build/index.html

echo "Shucky ducky."
