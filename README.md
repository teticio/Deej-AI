![](MBIT_logo.png)

# Deej-A.I.

[Robert Dargavel Smith](mailto:teticio@gmail.com) - Advanced Machine Learning end of Masters project ([MBIT School](http://mbitschool.academy), Madrid, Spain)

### Motivation

There are a number of automatic DJ tools around, which cleverly match the tempo of one song with another and mix the beats. To be honest, I have always found that kind of DJ rather boring: the better they are technically, the more it sounds just like one interminable song. In my book, it's not about how you play, but <i>what</i> you play. I have collected many rare records over the years and done a bit of deejaying on the radio and in clubs. I can almost instantly tell whether I am going to like a song or not just by listening to it for a few seconds. Or, if a song is playing, one that would go well with it usually comes to mind immediately. I thought that artificial intelligence could be applied to this "music intuition" as a music recommendation system based on simply *listening* to a song (and, of course, having an encyclopaedic knowledge of music).

Some years ago, the iPod had a very cool feature called *Genius*, which created a playlist on-the-fly based on a few example songs. Apple decided to remove this functionality (although it is still available in iTunes), presumably in a move to persuade people to subscribe to their music streaming service. Of course, Spotify now offers this functionality but, personally, I find the recommendations that it makes to be, at best, music I already know and, at worst, rather commercial and uncreative. I have a large library of music and I miss having a simple way to say "keep playing songs like this" (especially when I am driving) and something to help me discover new music, even within my own collection. I spent some time looking for an alternative solution but didn't find anything.

### Implementation details

A common approach is to use music genres to classify music, but I find this to be too simplistic and constraining. Is *Roxanne* by The Police reggae, pop or rock? And what about all the constantly evolving subdivisions of electronic music? I felt it necessary to find a higher dimensional, more continuous description of music and one that did not require labelling each track (i.e., an unsupervised learning approach).

The first thing I did was to [scrape](notebooks/Spewtify.ipynb) as many playlists from Spotify as possible. (Unfortunately, I had the idea to work on this after a [competition](https://labs.spotify.com/2018/05/30/introducing-the-million-playlist-dataset-and-recsys-challenge-2018/) to do something similar had already been closed, in which access to a million songs was granted.) The idea was that grouping by playlists would give some context or meaning to the individual songs - for example, "80s disco music" or "My favourite songs for the beach". People tend to make playlists of songs by similar artists, with a similar mood, style, genre or for a particular purpose (e.g., for a workout in the gym). Unfortunately, the Spotify API doesn't make it particularly easy to download playlists, so the method was rather crude: I searched for all the playlists with the letter 'a' in the name, the letter 'b', and so on, up to 'Z'. In this way, I managed to grab 240,000 playlists comprising 4 million unique songs. I deliberately excluded all playlists curated by Spotify as these were particularly commercial (I believe that artists can pay to feature in them).

Then, I created an embedding ("[Track2Vec](notebooks/Track2Vec.ipynb)") of these songs using the Word2Vec algorithm by considering each song as a "word" and each playlist as a "sentence". (If you can believe me, I had the same idea independently of [these guys](https://spandan-madan.github.io/Spotify/).) I found 100 dimensions to be a good size. Given a particular song, the model was able to convincingly suggest Spotify songs by the same artist or similar, or from the same period and genre. As the number of unique songs was huge, I limited the "vocabulary" to those which appeared in at least 10 playlists, leaving me with 450,000 tracks.

One nice thing about the Spotify API is that it provides a URL for most songs, which allows you to download a 30 second sample as an MP3. I downloaded all of these MP3s and converted them to a [Mel Spectrogram](notebooks/Get_spectrograms.ipynb) - a compact representation of each song, which supposedly reflects how the human ear responds to sound. In the same way as a human being can think of related music just by listening to a few seconds of a song, I thought that a window of just 5 seconds would be enough to capture the gist of a song. Even with such a limited representation, the zipped size of all the spectrograms came to 4.5 gigabytes!

The next step was to try to use the information gleaned from Spotify to extract features from the spectrograms in order to meaningfully relate them to each other. I trained a convolutional neural network to reproduce as closely as possible (in cosine proximity) the Track2Vec vector (output *y*) corresponding to a given spectrogram (input *x*). I tried both [one dimensional](notebooks/Speccy_1D.ipynb) (in the time axis) and [two dimensional](notebooks/Speccy_2D.ipynb) convolutional networks and compared the results to a baseline model. The baseline model tried to come up with the closest Track2Vec vector without actually listening to the music. This lead to a song that, in theory, everybody should either like (or hate) a little bit ;-) ([SBTRKT - Sanctuary](https://p.scdn.co/mp3-preview/5ac546c1bcbb1d0a6dbeced979dc95361ffc2530?cid=194086cb37be48ebb45b9ba4ce4c5936)), with a cosine proximity of 0.52. The best score I was able to obtain with the validation data before overfitting set in was 0.70. With a 300-dimensional embedding, the validation score was better, but so was that of the baseline: I felt it was more important to have a lower baseline score and a bigger difference between the two, reflecting a latent representation with more diversity and capacity for discrimination. The score, of course, is still very low, but it is not really reasonable to expect that a spectrogram can capture the similarities between songs that human beings group together based on cultural and historical factors. Also, some songs were quite badly represented by the 5 second window (for example, in the case of "Don't stop me now" by Queen, this section corresponded to Brian May's guitar solo...). I played around with an [Auto-Encoder](notebooks/Speccy_AE.ipynb) and a [Variational Auto-Encoder](notebooks/Speccy_VAE.ipynb) in the hope of forcing the internal latent representation of the spectrograms to be more continuous, disentangled and therefore meaningful. The initial results appeared to indicate that a two dimensional convolutional network is better at capturing the information contained in the spectrograms. I also considered training a Siamese network to directly compare two spectrograms. I've left these ideas for possible future research.

Finally, with a library of MP3 files, I mapped each MP3 to a series of Track2Vec vectors for each 5 second time slice. Most songs vary significantly from beginning to end and so the slice by slice recommendations are all over the place. In the same way as we can apply a Doc2Vec model to compare similar documents, I calculated a "Mp3ToVec" vector for each MP3, including each constituent Track2Vec vector according to its *TF-IDF* (Term Frequency, Inverse Document Frequency) weight. This scheme gives more importance to recommendations which are frequent *and* specific to a particular song. As this is an <img src="https://latex.codecogs.com/svg.latex?\Large&space;O(n^2)" title="\Large O(n^2)"/> algorithm, it was necessary to break the library of MP3s into batches of 100 (my 8,000 MP3s would have taken 10 days to process otherwise!). I checked that this had a negligible impact on the calculated vectors.

![](bokeh_plot.png)

### Results

You can see the some of the results at the end of this [workbook](notebooks/Deej-A.I.ipynb) and judge them for yourself. It is particularly good at recognizing classical music, spoken word, hip-hop and electronic music. In fact, I was so surprised by how well it worked, that I started to wonder how much was due to the TF-IDF algorithm and how much was due to the neural network. So I created another baseline model using the neural network with randomly initialized weights to map the spectrograms to vectors. I found that this baseline model was good at spotting genres and structurally similar songs, but, when in doubt, would propose something totally inappropriate. In these cases, the trained neural net seemed to choose something that had a similar energy, mood or instrumentation. In many ways, this was exactly what I was looking for: a creative approach that transcended rigid genre boundaries. By playing around with the <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon" title="\Large \epsilon"/> parameter which determines whether two vectors are the same or not, for the purposes of the TF-IDF algorithm, it is possible to find a good trade-off between the genre (global) and the "feel" (local) characteristics. I also compared the results to playlists generated by Genius in iTunes and, although it is very subjective, I felt that Genius was sticking to genres even if the songs didn't quite go together, and came up with less "inspired" choices. Perhaps a crowd sourced "Coca Cola" test is called for to be the final judge.

Certainly, given the limitations of data, computing power and time, I think that the results serve as a proof of concept.

### Applications

Apart from the original idea of an automatic (radio as opposed to club) DJ, there are several other interesting things you can do. For example, as the vector mapping is continuous, you can easily create a playlist which smoothly "[joins the dots](notebooks/Join_the_dots.ipynb)" between one song and another, passing through as many waypoints as you like. For example, you could travel from [soul to techno via funk and drum 'n' bass](https://soundcloud.com/teticio/mix-automatically-created-with-artificial-intelligence-deej-ai). Or from rock to opera :-).

Another simple idea is to listen to music using a microphone and to propose a set of next songs to play [on the fly](notebooks/Live.ipynb). Rather than comparing with the overall MP3ToVec, it might be more appropriate to just take into account the beginning of each song, so that the music segues more naturally from one track to another.

### Try it out for yourself

Once you have installed the required python packages with

```bash
pip install -r requirements.txt
```

and [downloaded](https://drive.google.com/file/d/1LM1WW1GCGKeFD1AAHS8ijNwahqH4r4xV/view?usp=sharing) the model weights to the directory where you have the python files, you can process your library of MP3s (and M4As). Simply run the following command and wait...


```bash
python MP3ToVec.py Pickles mp3tovec --scan c:/your_music_library
```

It will create a directory called "Pickles" and, within the subdirectory "mp3tovecs" a file called "mp3tovec.p". Once this has completed, you can try it out with

```bash
python Deej-A.I.py Pickles mp3tovec
```

Then go to [http://localhost:8050](http://localhost:8050) in your browser.  If you add the parameter `--demo 5`, you don't have to wait until the end of each song. Simply load an MP3 or M4A on which you wish to base the playlist; it doesn't necessarily have to be one from your music library. Finally, there are a couple of controls you can fiddle with (as it is currently programmed, these only take effect after the next song if one is already playing). "Keep on" determines the number of previous tracks to take into account in the generation of the playlist and "Drunk" specifies how much randomness to throw into the mix. Alternatively, you can create an MP3 mix of a musical journey of your choosing with

```bash
python Join_the_dots.py Pickles\mp3tovecs\mp3tovec.p tracks.txt mix.mp3 9
```

where "tracks.txt" is a textfile containing a list of MP3 or M4A files and, here, 9 is the number of additional tracks you want to generate between each of these.

If you are interested in the data I used to train the neural network, feel free to drop me an [email](mailto:teticio@gmail.com).
