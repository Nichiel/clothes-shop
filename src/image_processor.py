from transformers import AutoFeatureExtractor, AutoModel
import torchvision.transforms as T
import torch
import clip


class ImageProcessing:
    def __init__(self, model_ckpt: str) -> None:
        self.extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def encode_images(self, images: list) -> torch.Tensor:
        transformation_chain = T.Compose(
            [
                T.Resize(int((256 / 224) * self.extractor.size["height"])),
                T.CenterCrop(self.extractor.size["height"]),
                T.ToTensor(),
                T.Normalize(mean=self.extractor.image_mean, std=self.extractor.image_std),
            ]
        )

        image_batch_transformed = torch.stack([transformation_chain(image) for image in images])
        new_batch = {"pixel_values": image_batch_transformed.to(self.device)}
        with torch.no_grad():
            embeddings = self.model(**new_batch).last_hidden_state[:, 0].cpu()
        return embeddings


class ClipProcessing:
    def __init__(self, model_ckpt: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_ckpt, device=self.device)

    def encode_images(self, images: list) -> list:
        images_preprocessed = [self.preprocess(image).unsqueeze(0).to(self.device) for image in images]
        with torch.no_grad():
            image_features = [self.model.encode_image(image)[0] for image in images_preprocessed]
        return image_features

    def encode_text(self, text: str) -> torch.Tensor:
        text_tokenized = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokenized)
        return text_features
