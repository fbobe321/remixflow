import AVFoundation
import Foundation
import MLX

/// Decode an audio file to `[2, N]` float samples at 48 kHz (the model's rate),
/// and encode an MLXArray back to a WAV. Uses AVAudioConverter for resampling.
public enum AudioIO {

    /// Load `url` → MLXArray `[2, N]` @ 48 kHz stereo (mono is duplicated).
    public static func load(url: URL, sampleRate: Double = 48_000) throws -> MLXArray {
        let file = try AVAudioFile(forReading: url)
        let inFmt = file.processingFormat
        let outFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
                                   channels: 2, interleaved: false)!

        let inBuf = AVAudioPCMBuffer(pcmFormat: inFmt, frameCapacity: AVAudioFrameCount(file.length))!
        try file.read(into: inBuf)

        let ratio = sampleRate / inFmt.sampleRate
        let outCap = AVAudioFrameCount(Double(inBuf.frameLength) * ratio) + 1024
        let outBuf = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity: outCap)!

        let converter = AVAudioConverter(from: inFmt, to: outFmt)!
        var supplied = false
        var err: NSError?
        converter.convert(to: outBuf, error: &err) { _, status in
            if supplied { status.pointee = .noDataNow; return nil }
            supplied = true; status.pointee = .haveData; return inBuf
        }
        if let err { throw err }

        let n = Int(outBuf.frameLength)
        let ch = outBuf.floatChannelData!
        let left = Array(UnsafeBufferPointer(start: ch[0], count: n))
        let right = outFmt.channelCount > 1 ? Array(UnsafeBufferPointer(start: ch[1], count: n)) : left
        // [2, N]
        return MLX.stacked([MLXArray(left), MLXArray(right)], axis: 0)
    }

    /// Write MLXArray `[2, N]` @ `sampleRate` to a 16-bit WAV at `url`.
    public static func writeWav(_ samples: MLXArray, to url: URL, sampleRate: Double = 48_000) throws {
        let fmt = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2)!
        let file = try AVAudioFile(forWriting: url, settings: fmt.settings)
        let n = AVAudioFrameCount(samples.dim(1))
        let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: n)!
        buf.frameLength = n
        for c in 0..<2 {
            let row = samples[c].asType(.float32).asArray(Float.self)
            row.withUnsafeBufferPointer { buf.floatChannelData![c].update(from: $0.baseAddress!, count: Int(n)) }
        }
        try file.write(from: buf)
    }
}
