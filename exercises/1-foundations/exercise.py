
# Instructions
"""
You're a teaching assistant correcting student exams.
Keeping track of results manually is getting both tedious and mistake-prone.
You decide to make things a little more interesting by putting together some functions to count and calculate results for the class.
"""

## 1. Rounding Scores
"""
While you can give "partial credit" on exam questions, overall exam scores have to be `int`s.
So before you can do anything else with the class scores, you need to go through the grades and turn any `float` scores into `int`s. Lucky for you, Python has the built-in [`round()`][round] function you can use.

Create the function `round_scores(student_scores)` that takes a `list` of `student_scores`.
This function should _consume_ the input `list` and `return` a new list with all the scores converted to `int`s.
The order of the scores in the resulting `list` is not important.
"""
def round_scores(student_scores):
    rounded_scores = []
    for score in student_scores:
        rounded_scores.append(round(score))
    return rounded_scores

student_scores = [90.33, 40.5, 55.44, 70.05, 30.55, 25.45, 80.45, 95.3, 38.7, 40.3]
print(round_scores(student_scores))

## 2. Non-Passing Students
"""
As you were grading the exam, you noticed some students weren't performing as well as you had hoped.
But you were distracted, and forgot to note exactly _how many_ students.

Create the function `count_failed_students(student_scores)` that takes a `list` of `student_scores`.
This function should count up the number of students who don't have passing scores and return that count as an integer.
A student needs a score greater than **40** to achieve a passing grade on the exam.
"""
def count_failed_students(student_scores):
    count = 0
    for score in student_scores:
        if score < 40:
            count += 1
    return count
print(count_failed_students(student_scores))

## 3. The "Best"
"""
The teacher you're assisting wants to find the group of students who've performed "the best" on this exam.
What qualifies as "the best" fluctuates, so you need to find the student scores that are **greater than or equal to** the current threshold.

Create the function `above_threshold(student_scores)` taking `student_scores` (a `list` of grades), and `threshold` (the "top score" threshold) as parameters.
This function should return a `list` of all scores that are `>=` to `threshold`.
"""
def above_threshold(student_scores, threshold):
    result = [score for score in student_scores if score >= threshold]
    print(result)

above_threshold(student_scores=[90,40,55,70,30,68,70,75,83,96], threshold=75)

## 4. Calculating Letter Grades
"""
The teacher you are assisting likes to assign letter grades as well as numeric scores.
Since students rarely score 100 on an exam, the "letter grade" lower thresholds are calculated based on the highest score achieved, and increment evenly between the high score and the failing threshold of **<= 40**.

Create the function `letter_grades(highest)` that takes the "highest" score on the exam as an argument, and returns a `list` of lower score thresholds for each "American style" grade interval: `["D", "C", "B", "A"]`.
"""
def letter_grades(highest):
    thresholds = []
    increment = (highest - 40) / 4
    for i in range(4):
        threshold = int(41 + (i * increment))
        thresholds.append(threshold)
    return thresholds

letter_grades(highest=100)
letter_grades(highest=88)

## 5. Matching Names to Scores
"""
You have a list of exam scores in descending order, and another list of student names also sorted in descending order by their exam scores.
You would like to match each student name with their exam score and print out an overall class ranking.

Create the function `student_ranking(student_scores)` with parameters `student_scores` and `student_names`.
Match each student name on the student_names `list` with their score from the student_scores `list`.
You can assume each argument `list` will be sorted from highest score(er) to lowest score(er).
The function should return a `list` of strings with the format `<rank>. <student name>: <student score>`.
"""
def student_ranking(student_scores, student_names):
    ranking = []
    for i in range(len(student_scores)):
        rank = i + 1
        name = student_names[i]
        score = student_scores[i]
        ranking.append(f"{rank}. {name}: {score}")
    return ranking
student_scores = [100, 99, 90, 84, 66, 53, 47]
student_names =  ['Joci', 'Sara','Kora','Jan','John','Bern', 'Fred']
print(student_ranking(student_scores, student_names))

## 6. A "Perfect" Score
"""
Although a "perfect" score of 100 is rare on an exam, it is interesting to know if at least one student has achieved it.

Create the function `perfect_score(student_info)` with parameter `student_info`.
`student_info` is a `list` of lists containing the name and score of each student: `[["Charles", 90], ["Tony", 80]]`.
The function should `return` _the first_ `[<name>, <score>]` pair of the student who scored 100 on the exam.

If no 100 scores are found in `student_info`, an empty list `[]` should be returned.
"""
def perfect_score(student_info):
    for student in student_info:
        if student[1] == 100:
            return student
    return []

perfect_score(student_info=[["Charles", 90], ["Tony", 80], ["Alex", 100]])
["Alex", 100]

perfect_score(student_info=[["Charles", 90], ["Tony", 80]])
[]
