import torch
import torchvision.transforms as T
from emperic_algorithm_classify import prepare_image
from PIL import Image
import asyncio
from telebot.async_telebot import AsyncTeleBot
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_gpunet', pretrained=True, model_type='GPUNet-D2').to(device)
print('Model loaded')
model.load_state_dict(torch.load('GPUNet-D2_best_weights.pt'))
print('Weights loaded')
model.eval()

transforms_test = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return [round(i, 3) * 100 for i in (e_x / e_x.sum()).tolist()]


def predict_photo():
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  prepare_image('image.jpg', 'image_prepared.jpg')
  preds = []
  model.eval()
  image = Image.open('image_prepared.jpg').convert('RGB')
  transforms_test = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  image = transforms_test(image)[None, :, :, :].to(device)
  logit = model(image)
  logits = logit[0][:5].tolist()
  probas = softmax(np.array(logits))
  return get_preds_text(probas)

def get_preds_text(probas):
  probas = [round(i, 2) for i in probas]
  decoding = ['Downy Mildew', 'Bacterial Wilt', 'Fresh Leaf', 'Anthracnose', 'Gummy Stem Blight']
  x = zip(probas, decoding)
  x = sorted(x, key=lambda tup: tup[0])[::-1]
  result = 'Три возможные болезни:\n'
  for percent, illness in x[:3]:
    result += f'{illness} - вероятность {percent}%\n'
  return result

print('Bot is ready')
async def main():
    TOKEN = '7012079106:AAE64VoGoqvtJmNwdIicSP3xDT8DQjkQXBk'
    bot = AsyncTeleBot(TOKEN)
    # Bot adress: https://t.me/cucumber_diseases_bot

    @bot.message_handler(content_types=["photo"])
    async def predict(message):
        image_id = message.photo[len(message.photo) - 1].file_id
        file_path = (await bot.get_file(image_id)).file_path
        downloaded_file = await bot.download_file(file_path)
        with open("image.jpg", 'wb') as new_file:
            new_file.write(downloaded_file)
        await bot.send_message(message.chat.id, 'Фотография принята. Идет обработка')
        res = predict_photo()
        await bot.send_message(message.chat.id, res, parse_mode = 'Markdown')

    await bot.polling()
   
asyncio.run(main())