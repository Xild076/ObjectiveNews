import numpy as np
import nltk
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sentence_transformers import SentenceTransformer
from typing import Union, Type, List, Dict, Any
from nltk.stem import WordNetLemmatizer
from colorama import Fore, Style
from grouping import cluster_text
from article_analysis import cluster_articles, organize_clusters, provide_metrics


text_1 = """
PEARL HARBOR, Hawaii — Ira “Ike” Schab, a 104-year-old Pearl Harbor attack survivor, spent six weeks in physical therapy to build the strength to stand and salute during a remembrance ceremony honoring those killed in the Japanese bombing that thrust the U.S. into World War II some 83 years ago.

On Saturday, Schab gingerly rose from his wheelchair and raised his right hand, returning a salute delivered by sailors standing on a destroyer and a submarine passing by in the harbor.

“He’s been working hard because this is his goal,” said his daughter, Kimberlee Heinrichs, who traveled to Hawaii with Schab from their Beaverton, Oregon, home so they could attend the ceremony. “He wanted to be able to stand for that.”

Schab is one of only two servicemen who lived through the attack who made it to an annual remembrance ceremony hosted by the U.S. Navy and National Park Service on a grass field overlooking the harbor. A third survivor had been planning to join them but had to cancel because of health issues.

The Dec. 7, 1941 bombing killed more than 2,300 U.S. servicemen. Nearly half, or 1,177, were sailors and Marines on board the USS Arizona, which sank during the battle. The remains of more than 900 Arizona crew members are still entombed on the submerged vessel.

Dozens of survivors once joined the event but their attendance has declined as survivors have aged. Today there are only 16 still living, according to a list maintained by Kathleen Farley, the California state chair of the Sons and Daughters of Pearl Harbor Survivors. Military historian J. Michael Wenger has estimated there were some 87,000 military personnel on Oahu on the day of the attack.

Schab agreed when ceremony organizers asked him earlier this year to salute on behalf of all survivors and World War II veterans.

“I was honored to do it. I’m glad I was capable of standing up. I’m getting old, you know,” he said.

Schab was a sailor on the USS Dobbin at the time of the attack, the tuba player in the ship’s band. He had showered and put on a clean uniform when he heard the call for a fire rescue party.

He hurried topside to see Japanese planes flying overhead and the USS Utah capsizing. He quickly went back below deck to join a daisy chain of sailors feeding shells to an anti-aircraft gun topside.

Ken Stevens, 102, who served on the USS Whitney, joined Schab at the ceremony. USS Curtiss sailor Bob Fernandez, 100, had planned to attend but had to cancel due to health issues.

Ceremony attendees observed a moment of silence at 7:54 a.m., the same time the attack began eight decades ago. F-22 jets in missing man formation flew overhead shortly after.

Fernandez, speaking before the ceremony, recalled feeling shocked and surprised as the attack began.

“When those things go off like that, we didn’t know what’s what,” said Fernandez. “We didn’t even know we were in a war.”

Fernandez was a mess cook on the Curtiss and his job that morning was to bring sailors coffee and food as he waited tables during breakfast. Then they heard an alarm sound. Through a porthole, Fernandez saw a plane with the red ball insignia painted on Japanese aircraft fly by.

Fernandez rushed down three decks to a magazine room where he and other sailors waited for someone to unlock a door storing 5-inch, 38-caliber shells so they could begin passing them to the ship’s guns.

He has told interviewers over the years that some of his fellow sailors were praying and crying as they heard gunfire up above.

“I felt kind of scared because I didn’t know what the hell was going on,” Fernandez said.

The ship’s guns hit a Japanese plane that crashed into one of its cranes. Shortly after, its guns hit a dive bomber that then slammed into the ship and exploded below deck, setting the hangar and main decks on fire, according to the Navy History and Heritage Command.

Fernandez’s ship, the Curtiss, lost 21 men and nearly 60 of its sailors were injured.

Many laud Pearl Harbor survivors as heroes, but Fernandez doesn’t view himself that way.

“I’m not a hero,” he told The Associated Press in a phone interview from California, where he now lives with his nephew in Lodi. “I’m just nothing but an ammunition passer.”
"""

text_2 = """
South Korean President Yoon Suk Yeol apologized Saturday for the public anxiety caused by his short-lived attempt to impose martial law earlier this week, hours ahead of a parliamentary vote on impeaching him.

Yoon said in a brief televised address Saturday morning that he won't shirk legal or political responsibility for the declaration and promised not to make another attempt to impose it. He said he would leave it to his conservative political party to chart a course through the country's political turmoil, "including matters related to my term in office."

South Korean lawmakers are set to vote later Saturday on impeaching the president, as protests grew nationwide calling for his removal.

It wasn't immediately clear whether the motion submitted by opposition lawmakers would get the two-thirds majority required for Yoon to be impeached. But it appeared more likely after the leader of Yoon's own party on Friday called for suspending his constitutional powers, describing him as unfit to hold the office and capable of taking more extreme action, including renewed attempts to impose martial law.

Impeaching Yoon would require support from 200 of the National Assembly's 300 members. The opposition parties that jointly brought the impeachment motion have 192 seats combined.

That means they would need at least eight votes from Yoon's People Power Party. On Wednesday, 18 members of the PPP joined a vote that unanimously canceled martial law 190-0 less than three hours after Yoon declared the measure on television, calling the opposition-controlled parliament a "den of criminals" bogging down state affairs. The vote took place as hundreds of heavily-armed troops encircled the National Assembly in an attempt to disrupt the vote and possibly to detain key politicians.

Parliament said Saturday that it would meet at 5 p.m. local time. It will first vote on a bill appointing a special prosecutor to investigate influence peddling allegations surrounding Yoon's wife, and then on impeaching Yoon.

The turmoil resulting from Yoon's bizarre and poorly-thought-out stunt has paralyzed South Korean politics and sparked alarm among key diplomatic partners, including neighboring Japan and Seoul's top ally the United States, as one of the strongest democracies in Asia faces a political crisis that could unseat its leader.

Opposition lawmakers claim that Yoon's martial law declaration amounted to a self-coup and drafted the impeachment motion around rebellion charges.

The PPP decided to oppose impeachment at a lawmakers' meeting, despite pleas by its leader Han Dong-hun, who isn't a lawmaker and has no vote.

Following a party meeting on Friday, Han stressed the need to suspend Yoon's presidential duties and power swiftly, saying he "could potentially put the Republic of Korea and its citizens in great danger."

Han said he had received intelligence that during the brief period of martial law, Yoon ordered the country's defense counterintelligence commander to arrest and detain unspecified key politicians based on accusations of "anti-state activities."

Hong Jang-won, first deputy director of South Korea's National Intelligence Service, later told lawmakers in a closed-door briefing that Yoon called after imposing martial law and ordered him to help the defense counterintelligence unit to detain key politicians. The targeted politicians included Han, opposition leader Lee Jae-myung and National Assembly speaker Woo Won Shik, according to Kim Byung-kee, one of the lawmakers who attended the meeting.

The Defense Ministry said it had suspended the defense counterintelligence commander, Yeo In-hyung, who Han alleged had received orders from Yoon to detain the politicians. The ministry also suspended Lee Jin-woo, commander of the capital defense command, and Kwak Jong-geun, commander of the special warfare command, over their involvement in enforcing martial law.

Former Defense Minister Kim Yong Hyun, who has been accused of recommending Yoon to enforce martial law, has been placed under a travel ban and faces an investigation by prosecutors over rebellion charges.

Vice Defense Minister Kim Seon Ho, who became acting defense minister after Yoon accepted Kim Yong Hyun's resignation on Thursday, has testified to parliament that it was Kim Yong Hyun who ordered troops to be deployed to the National Assembly after Yoon imposed martial law.

"""

text_3 = """
Dec 7 (Reuters) - American content creators on TikTok asked followers to subscribe to their channels on rival platforms like Meta-owned (META.O), opens new tab Instagram and Alphabet's (GOOGL.O), opens new tab YouTube after a federal appeals court ruled that the social media app could be banned if it is not sold to a U.S.-based company by Jan. 19.
TikTok has become a major U.S. digital force as it has grown to 170 million U.S. users, especially younger people drawn to its short, often irreverent videos. It has sucked away advertisers from some of the largest U.S. players and added commerce platform TikTok Shop, which has become a marketplace for small businesses.
The U.S. Congress, fearing TikTok's Chinese owners are gathering information about American consumers, has passed a law requiring its owner, Chinese-backed ByteDance, to divest its TikTok in the U.S. or face a ban. On Friday, a federal appeals court upheld the law.
Threats by politicians and others to TikTok have been building for years, leading some users to brush off recent threats. That appeared to change on Friday, with the prospect of a ban in just six weeks. A Supreme Court appeal is still possible.
"For the first time I'm realizing that a lot of what I worked for could disappear," Chris Mowrey, a Democratic social media influencer with 470,000 TikTok followers, told Reuters. "I don't think it's been talked about enough how damaging it will be from an economic standpoint for small businesses and creators."
On the app, viewers and content creators voiced concerns and confusion, many saying they doubted the platform would survive, and that they were prepared for the worst.
Chris Burkett, a content creator on TikTok with 1.3 million people following his men's lifestyle videos, said he did not think the platform would last. "I don't think there's longevity on this app in the United States," he said in a video post, asking his audience to follow him on other social media platforms, such as Instagram, YouTube, X and Threads.
"We've put so many years and so much time into building our community here," said food travel content creator SnipingForDom, who has 898,000 followers on the app. While he did not think the end was near for TikTok, he still told followers to reach out to him on his Instagram page.
Others were also awaiting more information. Sarah Jannetti, a TikTok Shop consultant, said her clients are not worried about a potential TikTok ban and will not shift their businesses "until they see something that's more concrete."
"""

text = """
Israel's prime minister has announced its military has temporarily seized control of a demilitarized buffer zone in the Golan Heights, saying the 1974 disengagement agreement with Syria had "collapsed" with the rebel takeover of the country.

Benjamin Netanyahu said he had ordered the Israel Defense Forces (IDF) to enter the buffer zone and "commanding positions nearby" from the Israeli-occupied part of the Golan.

"We will not allow any hostile force to establish itself on our border," he said.

A UK-based war monitor said Syrian troops had left their positions in Quneitra province, part of which lies inside the buffer zone, on Saturday.

On Sunday, the IDF told residents of five Syrian villages inside the zone to stay in their homes until further notice.

The Golan Heights is a rocky plateau about 60km (40 miles) south-west of Damascus.

Israel seized the Golan from Syria in the closing stages of the 1967 Six-Day War and unilaterally annexed it in 1981. The move was not recognised internationally, although the US did so unilaterally in 2019.

The Israeli move in the buffer zone came after Syrian rebel fighters captured the capital, Damascus, and toppled Bashar al-Assad's regime. He and his father had been in power in the country since 1971.

Forces led by the Islamist opposition group Hayat Tahrir al-Sham (HTS) entered Damascus in the early hours of Sunday morning, before appearing on state television to declare Syria to now be "free".

Netanyahu said the collapse of the Assad regime was a "historic day in the Middle East".

"The collapse of the Assad regime, the tyranny in Damascus, offers great opportunity but also is fraught with significant dangers," he said.

He said events in Syria had been the result of Israeli strikes against Iran and the Iran-backed Lebanese armed group Hezbollah, Assad's allies, and insisted Israel would "send a hand of peace" to Syrians who wanted to live in peace with Israel.

The IDF seizure of Syrian positions in the buffer zone was a "temporary defensive position until a suitable arrangement is found", he said.

"If we can establish neighbourly relations and peaceful relations with the new forces emerging in Syria, that's our desire. But if we do not, we will do whatever it takes to defend the State of Israel and the border of Israel," he said.

After more than a year of war in the Middle East, Israel already has its hands full.

But the pace of events in Syria, it's northern neighbour, will be of real concern.

The IDF had already moved reinforcements to the occupied Golan.

In normal times, its warning to residents in several villages to stay in their homes because Israel would not hesitate to act if it felt it needed to would be seen as hugely provocative and enough to start a war.

Israel is especially concerned about who might get their hands on Bashar al-Assad's alleged arsenal of chemical weapons.

The leader of the Syrian rebellion is Abu Mohammed al-Jawlani. His family roots are in the occupied Golan Heights, where thousands of Israeli settlers now live alongside about 20,000 Syrians, most of them Druze, who stayed on after it was captured.

Israel will have no intention of giving that land up and is determined to protect its citizens.

During the 2011 Syrian uprising, Israel made the calculation that Assad, despite being an ally of both Iran and Hezbollah, was a better bet than what might follow his regime.

Israel will now be trying to calculate what comes next in Syria. Like everyone, it can only guess.

"""

clusters = organize_clusters(provide_metrics(cluster_articles('https://www.bbc.com/news/articles/cp9nxee2r0do', link_num=5)))

for i, cluster in enumerate(clusters):
    print(Fore.YELLOW + Style.BRIGHT + f"Group {i}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Representative Sentence:", Fore.RESET + str(cluster['representative']))
    print(Fore.GREEN + f"Summary:", Fore.RESET + str(cluster['summary']))
    print(Fore.BLUE + f"Sources:", Fore.RESET + str(cluster['sources']))
    print(Fore.BLUE + f"Reliability:", Fore.RESET + str(cluster['reliability']))
    print("-" * 80)


"""sentences = nltk.sent_tokenize(text_3)
print(Fore.RED + Style.BRIGHT + f"Number of sentences: {len(sentences)}" + Style.RESET_ALL)
result = cluster_text(sentences, context_weights={'single':0.25, 'context': 0.75}, score_weights={'sil': 0.45, 'db': 0.55, 'ch': 0.1}, clustering_method=AgglomerativeClustering, context=True, lemmatize=True, representative_context_len=1)
for cluster in result['clusters']:
    print(Fore.YELLOW + Style.BRIGHT + f"Cluster {cluster['cluster_id']}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Representative Sentence", Fore.RESET + cluster['representative'])
    print(Fore.BLUE + f"Representative with Context:", Fore.RESET + cluster['representative_with_context'])
    print("-" * 80)
print("Scores:", result['metrics'])
print(Fore.RED + Style.BRIGHT + f"Number of sentences: {len(sentences)}" + Style.RESET_ALL)
# Optimal values 12-7 8:05 PM: context_weights={'single':0.5, 'context': 0.5}, score_weights={'sil': 0.45, 'db': 0.55, 'ch': 0.1}
# Optimal values 12-8 11:06 AM: context_weights={'single':0.25, 'context': 0.75}, score_weights={'sil': 0.45, 'db': 0.55, 'ch': 0.1}, clustering_method=AgglomerativeClustering
"""
"""sentences_1 = nltk.sent_tokenize(text_1)
sentences_2 = nltk.sent_tokenize(text_2)
sentences_3 = nltk.sent_tokenize(text_3)

context_weights = {'single':0.5, 'context': 0.5}
score_weights = {'sil': 0.45, 'db': 0.55, 'ch': 0.1}
result_1 = cluster_text(sentences_1, context_weights=context_weights, score_weights=score_weights, clustering_method=AgglomerativeClustering, context=True, lemmatize=True, representative_context_len=1)
result_2 = cluster_text(sentences_2, context_weights=context_weights, score_weights=score_weights, clustering_method=AgglomerativeClustering, context=True, lemmatize=True, representative_context_len=1)
result_3 = cluster_text(sentences_3, context_weights=context_weights, score_weights=score_weights, clustering_method=AgglomerativeClustering, context=True, lemmatize=True, representative_context_len=1)

print(f"Weights: context={context_weights}, score={score_weights}")

print(f"Text 1: {len(sentences_1)} sentences, {len(result_1['clusters'])} clusters")
print(f"Text 2: {len(sentences_2)} sentences, {len(result_2['clusters'])} clusters")
print(f"Text 3: {len(sentences_3)} sentences, {len(result_3['clusters'])} clusters")
"""
"""result = cluster_articles('https://www.bbc.com/news/articles/cy8y7ggm89lo', link_num=10, debug_print=True)

for cluster in result['clusters']:
    print(Fore.YELLOW + Style.BRIGHT + f"Cluster {cluster['cluster_id']}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Representative Sentence:", Fore.RESET + str(cluster['representative']))
    print(Fore.GREEN + f"Sentences:", Fore.RESET + str(cluster['sentences']))
    print(Fore.BLUE + f"Sources:", Fore.RESET + str(cluster['sources']))
    print("-" * 80)
print("Scores:", result['metrics'])"""