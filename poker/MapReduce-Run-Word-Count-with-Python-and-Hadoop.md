---
title: 'MapReduce: Run Word Count with Python and Hadoop'
date: 2017-02-19 11:20:02
tags: 'python', 'hadoop', 'hortonworks', 'mapreduce'
---

## Install Hortonworks Sandbox
Hortonworks sandbox provides a nice playground for hadoop beginners to test their big data application.

- [Windows and Linux: Install Virtual Box](https://www.virtualbox.org/wiki/Downloads)
- [MAC: Install VMWare Fusion](https://www.vmware.com/go/downloadfusion)
- [Install Hortonworks Sandbox](https://hortonworks.com/downloads/)
- [Follow tutorial to setup environment and password](http://hortonworks.com/hadoop-tutorial/learning-the-ropes-of-the-hortonworks-sandbox/)
- The python shipped with hortonworks is Python 2.6, which is really old.
- Install Anaconda3 to upgrade python to Python 3.6 to default location /root/anaconda3
- [Anaconda](https://www.continuum.io/downloads)

```bash
bash Anaconda3-XXX-Linux-x86_64.sh
```

## Mapper
Code here is a modified version of mapper in [this blog](http://www.michael-noll.com/tutorials/writing-an-hadoop-mapreduce-program-in-python/)

```python
#!/root/anaconda3/bin/python
# Filename: word_count_mapper.py

import sys

# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    # split the line into words
    words = line.split()
    # increase counters
    for word in words:
        # write the results to STDOUT (standard output);
        # what we output here will be the input for the
        # Reduce step, i.e. the input for reducer.py
        #
        # tab-delimited; the trivial word count is 1
        print('{}\t{}'.format(word, 1))
```

## Reducer
Code here is a modified version of reducer in [this blog](http://www.michael-noll.com/tutorials/writing-an-hadoop-mapreduce-program-in-python/)

```python
#!/root/anaconda3/bin/python
# Filename: word_count_reducer.py

from operator import itemgetter
import sys

current_word = None
current_count = 0
word = None

# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()

    # parse the input we got from mapper.py
    word, count = line.split('\t', 1)

    # convert count (currently a string) to int
    try:
        count = int(count)
    except ValueError:
        # count was not a number, so silently
        # ignore/discard this line
        continue

    # this IF-switch only works because Hadoop sorts map output
    # by key (here: word) before it is passed to the reducer
    if current_word == word:
        current_count += count
    else:
        if current_word:
            # write result to STDOUT
            print('{}\t{}'.format(current_word, current_count))
        current_count = count
        current_word = word

# do not forget to output the last word if needed!
if current_word == word:
    print('{}\t{}'.format(current_word, current_count))

```
## Test Mapper and Reducer
- Download [Shakespeare](https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt) 
- Scp the txt into virtual machine, you can use scp on Mac and Linux ```scp -P 2222 t8.shakespeare.txt root@localhost:/root/ ```
- You can use [WinSCP](https://winscp.net/eng/download.php) to put the txt into virtual maxhine
- Make mapper and reducer executable ```chmod +x word_count_mapper.py word_count_reducer.py```
- **Test your mapper and reducer locally** ```cat t8.shakespear.txt | word_count_mapper.py | sort | word_count_reducer.py  ``` , this step is important because hadoop doesn't show the exact error output from python, so it's hard to debug python in hadoop.
- Upload the txt into hdfs under ```/demo/data```, using [Ambari file view](http://localhost:8080/#/main/views/FILES/1.0.0/AUTO_FILES_INSTANCE)
- Test mapper and reducer using hadoop

```bash
hadoop jar /usr/hdp/current/hadoop-mapreduce-client/hadoop-streaming.jar -file /root/mapper.py -mapper mapper.py -file /root/reducer.py -reducer reducer.py -input /demo/data/t8.shakespeare.txt -output /demo/output
```
- Clean up the output folder after the experiment, this step is important because hadoop will not overwrite existing folder

```bash
hdfs dfs -rm -r /demo/output
```

## Reference
- http://www.michael-noll.com/tutorials/writing-an-hadoop-mapreduce-program-in-python/