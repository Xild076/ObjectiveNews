import sys
import os
import streamlit as st
from notifypy import Notify

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def calculate_module_import_time(module_list:list):
    LOAD_TIMES = {
        'grouping': 4.76,
        'article_analysis': 6.05,
        'objectify': 5.35,
        'reliability': 4.12,
        'scraper': 4.33,
        'summarizer': 4.43,
        'synonym': 4.18,
        'utility': 4.09
    }
    module_list = list(set(module_list))
    
    final_time = 0

    for module in module_list:
        final_time += LOAD_TIMES[module]
    
    final_time -= LOAD_TIMES['utility'] * (len(module_list) - 1)
    return final_time

def estimate_time_taken_article(article_num):
    return 9.5 * article_num + 10

def estimate_time_taken_objectify(text):
    return 0.0075*len(text.split())+0.6

def make_notification(title:str, body:str):
    notif = Notify()
    notif.application_name = "Objective News"
    notif.title = title
    notif.icon = "data/images/Objective News.png"
    notif.message = body
    notif.audio = "data/sound/notif_sound.wav"
    notif.send()

