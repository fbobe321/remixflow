import Foundation
import Tokenizers
import MLX

/// Qwen2 tokenizer for prompts and lyrics, loaded from the model's `tokenizer/`
/// folder (tokenizer.json + config) via swift-transformers.
public final class RFTokenizer {
    private let tok: Tokenizer

    /// `folder` is the HF snapshot's `tokenizer/` directory.
    public init(folder: URL) async throws {
        self.tok = try await AutoTokenizer.from(modelFolder: folder)
    }

    public func encode(_ text: String) -> [Int] {
        tok.encode(text: text)
    }

    /// Token ids as an MLXArray [1, L] (Int32) ready for the text encoder.
    public func ids(_ text: String) -> MLXArray {
        let e = encode(text)
        let e2 = e.isEmpty ? [tok.bosTokenId ?? 0] : e   // never empty
        return MLXArray(e2.map { Int32($0) }).reshaped([1, e2.count])
    }
}
