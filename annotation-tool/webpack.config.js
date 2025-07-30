const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const MiniCssExtractPlugin = require("mini-css-extract-plugin");
const Dotenv = require("dotenv-webpack");
const webpack = require("webpack");
module.exports = {
  entry: "./src/index.tsx",
  output: {
    path: path.resolve(__dirname, "build"),
    filename: "[name].[contenthash].js",
    clean: true,
    publicPath: "/",
  },
  mode: "development",
  devtool: "source-map",
  resolve: {
    extensions: [".tsx", ".ts", ".js", ".jsx", ".mjs"],
    fullySpecified: false,
    alias: {
      react: path.resolve("./node_modules/react"),
      "react-dom": path.resolve("./node_modules/react-dom"),
    },
  },
  devServer: {
    historyApiFallback: true,
    static: "./dist",
    open: true,
    port: 3000,
    allowedHosts: ['.rtr.at']
  },
  module: {
    rules: [
      {
        test: /\.(ts|tsx)$/,
        use: "ts-loader",
        exclude: /node_modules/,
      },
      {
        test: /\.mjs$/,
        include: /node_modules/,
        type: "javascript/auto",
        resolve: {
          fullySpecified: false,
        },
      },
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: ["babel-loader"],
      },
      {
        test: /\.(css|scss)$/,
        use: [
          MiniCssExtractPlugin.loader, // or 'style-loader' if you prefer
          "css-loader",
          "sass-loader", // this gets ignored for plain `.css` files
        ],
      },
      {
        test: /\.(png|jpe?g|gif|svg)$/i,
        type: "asset/resource",
      },
    ],
  },
  plugins: [
    new MiniCssExtractPlugin(),
    new Dotenv(),
    new webpack.DefinePlugin({
      "process.env.PUBLIC_URL": JSON.stringify(process.env.PUBLIC_URL || ""),
      "process.env.HOST": JSON.stringify(process.env.HOST || ""),
    }),
    new HtmlWebpackPlugin({
      template: "./public/index.html",
      templateParameters: {
        PUBLIC_URL: process.env.PUBLIC_URL || "",
      },
    }),
  ],
};
