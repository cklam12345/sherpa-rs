use eyre::{bail, Result};
use sherpa_rs::{
    embedding_manager, speaker_id,
    vad::{Vad, VadConfig},
};
use std::env;

fn read_audio_file(path: &str) -> Result<(i32, Vec<f32>)> {
    let mut reader = hound::WavReader::open(path)?;
    let sample_rate = reader.spec().sample_rate as i32;

    if sample_rate != 16000 {
        bail!("The sample rate must be 16000.");
    }

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    Ok((sample_rate, samples))
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let enroll_file = &args[1];

    let (sample_rate, mut samples) = read_audio_file(enroll_file)?;

    for _ in 0..3 * sample_rate {
        samples.push(0.0);
    }

    let extractor_config = speaker_id::ExtractorConfig::new(
        "nemo_en_speakerverification_speakernet.onnx".into(),
        None,
        None,
        false,
    );

    let mut extractor = speaker_id::EmbeddingExtractor::new_from_config(extractor_config).unwrap();
    let mut embedding_manager =
        embedding_manager::EmbeddingManager::new(extractor.embedding_size.try_into().unwrap());

    let vad_model = "silero_vad.onnx".into();
    let window_size: usize = 512;
    let config = VadConfig::new(
        vad_model,
        0.4,
        0.4,
        0.5,
        sample_rate,
        window_size.try_into().unwrap(),
        None,
        None,
        Some(false),
    );

    let mut vad = Vad::new_from_config(config, 60.0 * 10.0).unwrap();
    let mut index = 0;

    while index + window_size <= samples.len() {
        let window = &samples[index..index + window_size];
        vad.accept_waveform(window.to_vec());

        if vad.is_speech() {
            while !vad.is_empty() {
                let segment = vad.front();
                let mut embedding =
                    extractor.compute_speaker_embedding(sample_rate, segment.samples)?;
                embedding_manager.add("enrolled_speaker".into(), &mut embedding)?;

                println!("Voice enrolled successfully!");
                vad.pop();
            }
        }

        index += window_size;
    }

    Ok(())
}
