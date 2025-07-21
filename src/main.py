from nltk import sent_tokenize
from grouping import cluster_texts, attention_model
import json
from rich import print

def split_sentences(text: str) -> list:
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]

text = """President Donald Trump defended Attorney General Pam Bondi over the weekend amid criticism from some in his base over a memo about accused sex trafficker Jeffrey Epstein, writing on Truth Social that Bondi is doing “a FANTASTIC JOB” in her role.

“What’s going on with my ‘boys’ and, in some cases, ‘gals?’ They’re all going after Attorney General Pam Bondi, who is doing a FANTASTIC JOB!” Trump wrote Saturday. “We’re on one Team, MAGA, and I don’t like what’s happening. We have a PERFECT Administration, THE TALK OF THE WORLD, and ‘selfish people’ are trying to hurt it, all over a guy who never dies, Jeffrey Epstein.”

Privately, Trump has also doubled down on his support for Bondi. The president called some of the attorney general’s most vocal critics over the weekend in an effort to stem the bleeding over the Epstein files, three sources familiar with the matter told CNN.

Trump’s calls included one with conservative activist Charlie Kirk on Saturday to express his support for Bondi. The call came as prominent MAGA supporters repeatedly criticized the attorney general at Kirk’s Turning Point USA Student Action Summit, a Florida event aimed at mobilizing young conservatives.

Members of the president’s inner circle have also reached out to some of Bondi’s critics to essentially ask them to ramp it down, noting that Trump, at this moment, was not getting rid of his attorney general. Sources cautioned that while Trump was currently still supporting Bondi, things could always change.

And on Sunday, Bondi appeared with Trump in the president’s box at the FIFA Club World Cup final in New Jersey. In one photo taken before the match, Trump could be seen flashing Bondi a thumbs-up.

President Donald Trump gives a thumbs-up to Attorney General Pam Bondi at the FIFA Club World Cup 2025 final football match in East Rutherford, New Jersey, on Sunday.
President Donald Trump gives a thumbs-up to Attorney General Pam Bondi at the FIFA Club World Cup 2025 final football match in East Rutherford, New Jersey, on Sunday. Charly Triballeau/AFP/Getty Images
CNN reported Friday that Deputy FBI Director Dan Bongino has told people he is considering resigning amid a major clash between the FBI and the Department of Justice over the fallout from the release of the memo, which concluded there is no evidence Epstein kept a “client list” or was murdered.

Trump said “I think so” when asked by reporters Sunday whether Bongino was still FBI deputy director but indicated “he’s in good shape” after speaking with him earlier in the day.

“I think so. I spoke to him today. Dan Bongino is a very good guy. I’ve known him a long time. … And he sounded terrific,” he said, adding, “I think he’s in good shape.”

The memo’s release last week prompted fierce blowback from some Trump allies, including right-wing activist Laura Loomer, who has called for Bondi’s ouster.

Former Trump adviser Steve Bannon dedicated much of his Monday “War Room” podcast to the memo, questioning the administration’s dedication to transparency. Bannon later argued to CNN the federal investigation appears to have been mismanaged.

A Trump adviser called the memo’s release a “political nightmare” and suggested it could’ve been published before the holiday weekend when fewer people might see it — or perhaps even after the 2026 midterm elections.

Former Fox News host Tucker Carlson argued on his podcast that Bondi is “covering up crimes, very serious crimes by their own description.”

Dan Bongino during a forum at the Heritage Foundation in Washington, DC, on August 26, 2024.
Related article
Deputy FBI Director Bongino has told people he is considering resigning amid Epstein files fallout, sources say

Epstein was a disgraced financier and convicted sex offender whose criminal case has long captured significant public attention, in part because of his ties to wealthy and high-profile people. In August 2019, while he was awaiting trial in a federal criminal case, Epstein was found unresponsive in his New York City jail cell. He was taken to a hospital, where he was pronounced dead. His death was ruled a suicide.

The death, though, was heavily scrutinized, and during his 2024 campaign, Trump said he would consider releasing additional government files on the case. Many of the president’s supporters hoped that release would implicate other high-profile figures, or undercut the notion that Epstein killed himself. But the Justice Department announced in a memo Monday that there was no evidence he kept a “client list” or was murdered, fueling rage and suspicion among many in MAGA world.

Trump himself shrugged off questions about the investigation into Epstein and the memo’s release, telling reporters at the White House on Tuesday, “I can’t believe you’re asking a question on Epstein at a time like this.”

The president repeated his frustration in his Saturday post, writing, “For years, it’s Epstein, over and over again,” while accusing a slew of political adversaries, including former President Barack Obama, former Secretary of State Hillary Clinton, former FBI Director James Comey, former CIA Director James Brennan “and the Losers and Criminals of the Biden Administration” who he claimed “created the Epstein Files.”

Jeffrey Epstein in Cambridge, Massachusetts in 2004.
Related article
What to know about the Jeffrey Epstein saga

“Why didn’t these Radical Left Lunatics release the Epstein Files? If there was ANYTHING in there that could have hurt the MAGA Movement, why didn’t they use it?” he said.

But the federal investigation of Epstein that led to his indictment happened during Trump’s first term. And Epstein’s suicide in federal prison also occurred during the first Trump administration. Then-Attorney General Bill Barr personally looked at video to make sure there wasn’t evidence of foul play, and the Department of Justice determined Epstein died by suicide.

And Trump urged FBI Director Kash Patel to turn his attention to the president’s own priorities, writing, “Kash Patel, and the FBI, must be focused on investigating Voter Fraud, Political Corruption, ActBlue, The Rigged and Stolen Election of 2020, and arresting Thugs and Criminals, instead of spending month after month looking at nothing but the same old, Radical Left inspired Documents on Jeffrey Epstein.”

“LET PAM BONDI DO HER JOB — SHE’S GREAT!” he added, calling Epstein “somebody that nobody cares about.”

This story has been updated with further details.

CNN’s Kaitlan Collins, Hannah Rabinowitz, Alayna Treene and Evan Perez contributed to this report.
”"""

"""def print_clusters_formatted(sentences, clusters, title):
    print(f"\n{title}")
    print("=" * len(title))
    
    if not clusters:
        print("No clusters found")
        return
    
    cluster_map = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = []
        cluster_map[cluster_id].append((i, sentences[i]))
    
    for cluster_id in sorted(cluster_map.keys()):
        print(f"\nCluster {cluster_id} ({len(cluster_map[cluster_id])} sentences):")
        print("-" * 50)
        for idx, sentence in cluster_map[cluster_id]:
            print(f"  [{idx}] {sentence}")
    
    print(f"\nTotal clusters: {len(cluster_map)}")
    print(f"Total sentences: {len(sentences)}")

sentences = split_sentences(text)
print(f"Extracted {len(sentences)} sentences from text")

clusters1 = cluster_texts(sentences, att_model=attention_model, weights=0.4088074527809119, context=True, context_len=2, preprocess=False, norm='max', n_neighbors=6, n_components=4, umap_metric='correlation', cluster_metric='euclidean', algorithm='prims_kdtree', cluster_selection_method='leaf')
print_clusters_formatted(sentences, clusters1, "CLUSTERING 1 MODEL")

clusters2 = cluster_texts(sentences, att_model=None, weights=0.4088074527809119, context=True, context_len=2, preprocess=False, norm='max', n_neighbors=6, n_components=4, umap_metric='correlation', cluster_metric='euclidean', algorithm='prims_kdtree', cluster_selection_method='leaf')
print_clusters_formatted(sentences, clusters2, "CLUSTERING 2 MODEL")

clusters3 = cluster_texts(sentences)
print_clusters_formatted(sentences, clusters3, "CLUSTERING 3 MODEL")"""


from utility import clean_text, SentenceHolder

sentences = split_sentences(text)
sentence_holders = [SentenceHolder(s) for s in sentences]

def print_clusters_pretty(clusters):
    for cluster in clusters:
        print(f"\nCluster {cluster['label']} ({len(cluster['sentences'])} sentences):")
        print("-" * 50)
        for idx, sent_holder in enumerate(cluster['sentences']):
            print(f"  [{idx}] {getattr(sent_holder, 'text', str(sent_holder))}")
        rep = cluster.get('representative')
        if rep is not None:
            print(f"\n  Representative: {getattr(rep, 'text', str(rep))}")
        print()

clusters = cluster_texts(sentence_holders, {})
print_clusters_pretty(clusters)