import torch
import torchvision.transforms as T
from emperic_algorithm_classify import prepare_image
from PIL import Image
import asyncio
from telebot.async_telebot import AsyncTeleBot
import numpy as np
from datetime import datetime


device = torch.device('cpu')

model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_gpunet', pretrained=True, model_type='GPUNet-D2').to(device)
print('Model loaded')
model.load_state_dict(torch.load('GPUNet-D2_best_weights.pt', map_location=torch.device('cpu')))
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
  device = torch.device('cpu')
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

translation = dict()
translation['Downy Mildew'] = 'Ложная мучнистая росса'
translation['Bacterial Wilt'] = 'Бактериальное увядание стебля'
translation['Fresh Leaf'] = 'Здоровое растение'
translation['Anthracnose'] = 'Аскохитоз'
translation['Gummy Stem Blight'] = 'Липкий ожог стебля'

def get_preds_text(probas):
  probas = [round(i, 2) for i in probas]
  decoding = ['Downy Mildew', 'Bacterial Wilt', 'Fresh Leaf', 'Anthracnose', 'Gummy Stem Blight']
  x = zip(probas, decoding)
  x = sorted(x, key=lambda tup: tup[0])[::-1]
  result = 'Вероятности трех возможных классов:\n'
  for percent, illness in x[:3]:
    result += f'{translation[illness]} - {str(percent)}%\n'
  print(x)
  return result, x[0][1]

text_for_start = 'Здравствуйте! В это боте вы можете исследовать свой лист на зболевания. Просто отправьте фотографию с растением. Просто отправьте фотографию с листом рпстения. Обратите внимание, что распознавание будет намного более качественным, если исследуемый лист будет изображен на фотографии крупным планом.'
text_for_sending = 'Пожалуйста, отправьте фотографию с листом растения. Обратите внимание, что распознавание будет намного более качественным, если исследуемый лист будет изображен на фотографии крупным планом.'
therapy_dict = dict()
therapy_dict['Fresh Leaf'] = ''
therapy_dict['Downy Mildew'] = '[Ссылка](https://www.letto.ru/blog/ogorod/profilaktika_i_lechenie_bolezni_ogurtsov_peronosporoz/) с рекомендуемым способом лечения ложной мучнистой россы'
therapy_dict['Bacterial Wilt'] = '[Ссылка](https://dacha.avgust.com/for-garden-home/directory-page/bakterialnoe-i-trakheomikoznoe-uvyadanie/) с рекомендуемым способом лечения бактериального увядания'
therapy_dict['Anthracnose'] = '[Ссылка](https://www.pesticidy.ru/%D0%90%D1%81%D0%BA%D0%BE%D1%85%D0%B8%D1%82%D0%BE%D0%B7_%D0%BE%D0%B3%D1%83%D1%80%D1%86%D0%B0) с рекомендуемым способом лечения Аскохитоза'
therapy_dict['Gummy Stem Blight'] = '[Ссылка](https://cucurbitbreeding.wordpress.ncsu.edu/watermelon-breeding/nc-state-watermelon-disease-handbook/gummy-stem-blight-didymella-bryoniae/) с рекомендуемым способом лечения липокого ожога стебля'

print('Bot is ready')
async def main():
    TOKEN = ''
    bot = AsyncTeleBot(TOKEN)

    @bot.message_handler(content_types=["photo"])
    async def predict(message):
        image_id = message.photo[len(message.photo) - 1].file_id
        file_path = (await bot.get_file(image_id)).file_path
        downloaded_file = await bot.download_file(file_path)
        with open("image.jpg", 'wb') as new_file:
            new_file.write(downloaded_file)
        now = datetime.now()
        cur_time = now.strftime("%Y-%m-%d-%H:%M:%S")

        if message.from_user.username != '':
            with open(f"leaves_data/@{message.from_user.username}_{cur_time}.jpg", 'wb') as new_file:
                new_file.write(downloaded_file)
        else:
            with open(f"leaves_data/{message.chat.id}_{cur_time}.jpg", 'wb') as new_file:
                new_file.write(downloaded_file)
        await bot.send_message(message.chat.id, 'Фотография принята. Идет обработка')
        res, most_possible_illness = predict_photo()
        await bot.send_message(message.chat.id, res, parse_mode='Markdown')
        if most_possible_illness != 'Fresh Leaf':
           therapy = therapy_dict[most_possible_illness]
           print(therapy)
           await bot.send_message(message.chat.id, therapy, parse_mode='Markdown')
    
    @bot.message_handler(commands=['start'])
    async def hello_message(message):
        await bot.send_message(message.chat.id, text_for_start, parse_mode='Markdown')

    @bot.message_handler(content_types=["text"])
    async def msg(message):
        await bot.send_message(message.chat.id, text_for_sending)

    @bot.message_handler(commands=['start'])
    async def hello_message(message):
        await bot.send_nessage(message.chat.id, text_for_start)
    
    await bot.polling()
   
asyncio.run(main())