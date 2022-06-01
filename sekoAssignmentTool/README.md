# The SEKO Assignment: Efficient and Fair Assignment of Students to Multiple Seminars 

This Python script implements the SEKO assignment strategy which is a lottery procedure for the fair assignment of interested students to seminars. The seminars are often limited in terms of the maximum number of participants. Due to the place limitation, those seminars are often offered several times to serve the students' demands. Still, some seminars are more popular than others and it may not be possible to grant access to all interested students due to capacity limitations. The SEKO assignment strategy achieves the following goals: 
1. Efficiency by utilizing all available seminar places, 
2. Satisfying all students by trying to assign at least one seminar to each student, and 
3. Fairness by considering the number of assigned seminars per student. 

The limited capacity (number of places per seminar) is taken into account in the lottery. Furthermore, the lottery procedure takes into account how many seminars a participant has already been assigned. The previous maximum number of assigned seminars is used for this: `curMax`. A participant `i` has already been assigned `yi` seminars. Then the (unnormalized) probability that this participant will be assigned a seminar place for the next seminar is calculated as follows:

`pi = curMax+1-yi`

If a student `i` has not yet been assigned a seminar, this will be taken into account separately and the probability increased accordingly. This means that the number of participants with at least one assigned seminar (ALOS) is maximized.

`pi = pi*100 = (curMax+1)*100`

The probabilities across all students are then normalized: `p = p/p.sum()`

_Example: Some seminars have already been assigned. Seminar 1 is next:_

    user 19 has 2 seminars assigned: [5 8] and requested [0 1 2 4 5 6 8]; prob. for next seminar 0.07%
    user 22 has 1 seminars assigned: 6 and requested [0 1 3 4 6]; prob. for next seminar 0.13%
    user 23 has 0 seminars assigned: [] and requested [0 3 4]; prob. for next seminar 19.71%

The probabilities in the assignment rounds are as follows. `p_0` is the probability for participants who have not yet been assigned a seminar. `p_i` is the probability for participants who have previously been assigned `i` seminars. `p_*=100%` indicates that there are fewer requests than seminar places and all students will be assigned to the seminar in that raound. The number of assigned participants is given in brackets.
	
    Round 0: Seminar "Python for Beginners A" p_*=100.0000% (8);
    Round 1: Seminar "Python for Beginners B" p_*=100.0000% (8);
    Round 2: Seminar "Online teaching" p_*=100.0000% (8);
    Round 3: Seminar "Integer Programming" p_*=100.0000% (9);
    Round 4: Seminar "Random numbers" p_*=100.0000% (11);
    Round 5: Seminar "Sustainability A" p_0=19.7109% (5); p_1=0.1314% (9); p_2=0.0657% (4);
    Round 6: Seminar "Sustainability B" p_0=16.3867% (6); p_1=0.1229% (9); p_2=0.0819% (6); p_3=0.0410% (2);
    Round 7: Seminar "MIP 1" p_0=31.7662% (3); p_1=0.2541% (12); p_2=0.1906% (6); p_3=0.1271% (3); p_4=0.0635% (2);
    Round 8: Seminar "MIP 2" p_0=83.8926% (1); p_1=0.6711% (12); p_2=0.5034% (13); p_3=0.3356% (3); p_4=0.1678% (3);
    Round 9: Seminar "MIP 3" p_1=5.1948% (8); p_2=3.8961% (13); p_3=2.5974% (2); p_4=1.2987% (2); 	
	
During the assignment, the seminars are processed in ascending order of their popularity. The seminar with the lowest demand is started first. If a seminar has more available places than requests, all participants will be assigned to the seminar.

This assignment strategy ensures that the available seminar places are used as well as possible and that the maximum number of seminar places is allocated to the seminar participants. The lottery procedure takes into account a maximum number of seminars that are assigned to a seminar participant; the default value is 999 seminars. The expected fairness of the procedure is close to the theoretical optimum. Furthermore, the number of participants with at least one assigned seminar is maximized. The SEKO assignment strategy achieves the objectives mentioned above.

## Input file
The input data is specified in an Excel file (see 'input.xlsx') and must be in the Excel sheet "registration". The columns contain the available seminars. In the sample file, these are the following 10 seminars:
1. Python for Beginners A	
2. Python for Beginners B	
3. Online teaching	
4. Integer Programming	
5. Random numbers	
6. Sustainability A	
7. Sustainability B	
8. MIP 1	
9. MIP 2	
10. MIP 3

There is one line for each registered student. A "1" indicates that the student has registered for the seminar. In the sample file, Mia has signed up for 4 seminars: Python for Beginners A, Python for Beginners B, Online teaching, MIP 2.

Some seminars are offered at different times, such as "Python for Beginners A" and "Python for Beginners B". To indicate that seminars are identical in content, there is an extra row "seminar_type" with which seminars with identical content are indicated. In this case, Mia would be assigned to at most one of the two Python seminars, but never to both Python seminars.

In order to indicate the number of places available per seminar, there is an extra line "places". The number of places per seminar is specified individually per seminar. 

_Example input file: input.xlsx:_

Person | Python for Beginners A | Python for Beginners B | Online teaching | Integer Programming | Random numbers | Sustainability A | Sustainability B | MIP 1 | MIP 2 | MIP 3
-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----
seminar_type |  1 | 	1 | 2 | 3 | 4 | 5 | 5 | 6 | 6 | 6
capacity  | 17 | 15 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 10
Mia | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0
Emma | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0
Hannah | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0

## Execution of the program
The program is called in a windows terminal console. Various parameters can be specified here. Important is the `SEED`, which initializes the [random number generator](https://de.wikipedia.org/wiki/Random Number Generator) for the lottery procedure. Every time the lottery process starts with the same seed, the same random number sequence is generated. This is important for reproducibility. Hence, the lottery procedure can also be traced and reproduced afterwards. The random numbers generated are used to assign the seminar participants according to the probabilities (see above).



```
usage: sekoAssignmentTool.exe [-h] [-s SEED] [-i INPUT] [-o OUTPUT] [-m MAXIMUM] [-v [VERBOSE]]

The SEKO Assignment: Efficient and Fair Assignment of Students to Seminars

optional arguments:
  -h, --help            show this help message and exit
  -s SEED, --seed SEED  randomly initialize the random number generator with that seed. Default: current time stamp
  -i INPUT, --input INPUT
                        Name of the input-file which is a standard excel file. Default: input.xlsx
  -o OUTPUT, --output OUTPUT
                        Output of the SEKO assignment will be saved in this OUTPUT excel file. Default: output.xlsx
  -m MAXIMUM, --maximum MAXIMUM
                        Maximum number of seminars which is assigned to a student. Default: 999
  -v [VERBOSE], --verbose [VERBOSE]
                        Prints the assignment steps verbosely while iterating over the seminars. Default: True
```

## Output of the program
The program saves the result of the assignment in an Excel file. The default name is `output.xlsx`, but any file name can be specified. Various Excel sheets are available to show the result of the lottery.

#### Sheet: Seminar
There is one line for each seminar offered. The _capacity_ column shows the number of available seminar places for this seminar, i.e. the entry from the input file. The _assigned_students_ column indicates how many students have been assigned to this seminar. The _requests_ column shows how many students have registered for the seminar. The _seminar_utilization_ column indicates the percentage of seminar places assigned with respect to the seminar's capacity. The following columns (1, 2, 3, etc.) contain the names of the participants determined by lottery.

#### Sheet: Assignment
Das Sheet _Assignment_ ist wie die Eingabedatei aufgebaut. In den Spalten finden sich die Seminare; jede Zeile spiegelt die Zuweisung für eine Person wider. Eine `1` bedeutet, dass die Person in der entsprechenden Zeile als TeilnehmerIn für das Seminar in der entsprechenden Spalte ausgewürfelt wurde.

#### Sheet: Difference
The _Difference_ sheet is structured like the input file. The columns contain the seminars; each row reflects the assignment for one person. `1` means that the student in the corresponding row was assigned as a participant for the seminar in the corresponding column. `-1` means that the student in the corresponding line had registered for the seminar, but the participant was not given a place (due to the limitation of the number of places per seminar or the maximum number of seminars per participant). `0` means that the student did not request the seminar.

#### Sheet: Stats_Person
The sheet _Stats_Person_ provides statistics from the students' persepctive. The _requested_ column shows how many seminars the student has registered for. The _assigned_ column indicates how many seminars were assigned. The _ratio_ column indicates the percentage of how many seminars were assigned relative to the registered ones.

#### Sheet: Parameters
The input parameters are recorded in this sheet. In particular, the `seed` for the random number generator is saved here to ensure reproducibility.

#### Sheet: Waiting_places
Finally, for each seminar the order list of students (waiting places) are additionally provided. In case, student assigned to a seminar cannot attend, then the waiting list can be used in FIFO manner to fill the seminar places.