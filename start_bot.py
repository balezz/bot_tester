# coding: utf-8

import telebot
from qa_model import QAModel
from rubert import RuBertWrapper

CONFIG_PATH = 'models/rubert_cased_L-12_H-768_A-12_pt'
rubert = RuBertWrapper(CONFIG_PATH)

QA_PATH = 'questions_answers/qa_small.txt'
qai = QAModel(QA_PATH, rubert)

TOKEN = '1409080754:AAHRHF35y-BwFPTF6Mb5REuxILvAg0e0gM4'
bot = telebot.TeleBot(TOKEN)

START_MSG = 'Предлагаем Вам ответить на несколько вопросов по системному анализу.'


@bot.message_handler(commands=['start'])
def handle_start(message):
    qai.idx = 0
    qai.session_active = True
    cid = message.chat.id
    bot.send_message(cid, START_MSG)
    first_question = qai.questions[qai.idx]
    bot.send_message(cid, first_question)


@bot.message_handler(func=lambda message: True, content_types=['text'])
def process_answer(m):
    if not qai.session_active:
        bot.send_message(m.chat.id, "Для начала опроса отправьте команду /start.\n")
        return
    qai.user_answers[qai.idx] = m.text
    qai.idx += 1
    if qai.idx < len(qai.questions):
        bot.send_message(m.chat.id, "Ответ принят. Следующий вопрос.\n")
        next_question = qai.questions[qai.idx]
        bot.send_message(m.chat.id, next_question)
    else:
        user_name = m.from_user.username
        good_buy(m.chat.id, user_name)


def good_buy(chat_id, user_name):
    score = qai.calculate_score()
    dist_str = ''
    for d in qai.distances:
        dist_str += '{:.2f} '.format(float(d))
    qai.session_active = False
    bot.send_message(chat_id, f"Опрос закончен. \n Расстояния эмбеддингов: {dist_str} \n Итоговый результат: {pretty(score)}.")
    qai.save_results(user_name)


def pretty(score):
    score = int(score)
    if score == 1:
        return '1 балл'
    elif score in (2, 3, 4):
        return f'{score} балла'
    else:
        return f'{score} баллов'


if __name__ == '__main__':
    bot.polling()
