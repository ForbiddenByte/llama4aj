# llama4aj

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## About

A Direct Android & Java Build For The [mybigday](https://github.com/mybigday) Serverless Server-like Chat Completion Implementation

Allows For Fast Inference & Intuitive App Building - Checkout The Example Apps & Try Building Them Yourself!

## Getting Started

```
./gradlew :examples:android-app:build # You can just copy paste the .apk file to your phone and install it.
# Or installDebug If You Have adb
#
# Or For The More Simple Desktop App
# mv model.gguf examples/desktop-app/ - Place The Model inside examples/desktop-app/
./gradlew :examples:desktop-app:run
```

***COMING TO MAVEN SOON***

***SCALA VERSION COMING SOON***

## TODO

A lot more configuring to make sure it is alligned with upstream [llama.rn](https://github.com/mybigday/llama.rn)

Proper bootstrap system - The way this is currently setup is kinda terrible as it's a combination of my manual setup and an attempt to setup a bootstrap system. Which is already out-of-sync with upstream so I need it to be more robust. 

Proper syncing system - This bootstrap system could and should be used more like a syncing system..

Cleanup...

Tests

Docs

Videos

Cleanup........

[mybigday]:(https://github.com/mybigday)
