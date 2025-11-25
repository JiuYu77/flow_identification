#!/bin/bash

rm -r server/dist
npm run build
mv dist server
