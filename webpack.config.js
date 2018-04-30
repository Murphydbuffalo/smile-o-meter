module.exports = {
  mode: 'production',
  entry: './javascripts/index.js',
  output: {
    path: '/Users/danielmurphy/Code/ML/smile-o-meter/build',
    filename: 'index-[hash].min.js'
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['env']
          }
        }
      },
      {
        test: /\.scss$/,
        use: {
          loader: 'sass-loader'
        }
      }
    ]
  }
};
