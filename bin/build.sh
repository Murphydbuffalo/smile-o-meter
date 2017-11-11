#!/bin/sh

# Exit the script if any command within it returns with an error.
set -e

rm -rf ./build/*
cp -R ./favicons/* ./images ./build

# Concatenate and minify all files in the `/stylesheets` and `/javascript` directories.
# And move the resulting manifest files to the `/build` directory of the project.
#
# We make separate calls to `uglify` and `minify` (which is just a wrapper library
# that can call `uglify` if you want), because we are using the `harmony` branch
# of `uglify`. This allows us to minify ES6 JavaScript.
#
# We also give each manifest file a unique name that includes the hash of the file
# contents of the corresponding file (either `index.css` and `index.js`).
# These unique names, combined with setting the `Cache-Control` header's `maxage`
# value allow browsers to avoid making requests for our JS and CSS files for up
# to one year, so long as the contents of `index.css/js` haven't changed.
JS_HASH=`md5sum ./javascripts/index.js | awk '{print $1}'`
CSS_HASH=`md5sum ./stylesheets/index.css | awk '{print $1}'`
uglifyjs ./javascripts/index.js --compress --mangle toplevel --output "./build/index-$JS_HASH.min.js"
uglifycss --output "./build/index-$CSS_HASH.min.css" ./stylesheets/normalize.css ./stylesheets/skeleton.css ./stylesheets/index.css

# Build full HTML file from `template.html`
#
# Replace the CSS and JS file names in `template.html` with the hash-based filenames for our CSS and JS manifest files.
# And write the resulting HTML to `template_with_manifests.html`.
JS_FILENAME=`find ./build -name "index-*.min.js" -exec basename {} \;`
CSS_FILENAME=`find ./build -name "index-*.min.css" -exec basename {} \;`
sed "s/index.min.js/$JS_FILENAME/" ./template.html > ./build/index.html
sed -i "" "s/index.min.css/$CSS_FILENAME/" ./build/index.html

echo "Shucky ducky."
