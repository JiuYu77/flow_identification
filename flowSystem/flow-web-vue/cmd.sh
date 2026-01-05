#!/bin/bash

server=../server

rm -r $server/dist
npm run build
mv dist $server
