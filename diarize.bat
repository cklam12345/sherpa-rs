@echo off
setlocal

:: Define variables
set "diarize=%USERPROFILE%\wave1.0\sherpa-rs"

echo diarize incoming...
cd %diarize%
cargo run --example diarize_whisper %1 > %2