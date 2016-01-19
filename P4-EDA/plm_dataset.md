## Bundesliga data for fantasy football
#### A dataset containing 10,122 observations of Bundesliga players from playlivemanager.com

Data has been gathered in JSON format via the API, i.e. at http://www.playlivemanager.com/api/players/64440/stats/round (where 64440 is the respective player id). The JSON data has been converted to a table and information such as home club, position and player name has been matched via webscraping as far as possible from www.playlivemanager.com.

Scraper and conversion can be found at https://github.com/bethe/livemanagerstats

The dataset contains 1 entry for every player (598) and every match played so far (17). There are 32 variables:

#### Variables
##  $ X                 : int  1 2 3 4 5 6 7 8 9 10 ... Index (1 -- 10166)
##  $ assist            : int  NA NA NA NA NA NA NA NA NA NA ... Assists to a goal in a match (NA, 0 -- 3)
##  $ clean_sheet       : int  0 0 0 0 1 0 0 1 0 0 ... Clean Sheets (Keeper only) (0, 1)
##  $ goal              : int  NA NA NA NA NA NA NA NA NA NA ... Goals scored (0 -- 5)
##  $ matchday          : int  1 2 3 4 5 6 7 8 9 10 ... Index of matches (1 -- 17)
##  $ game_play_duration: int  90 90 90 90 90 90 90 90 90 90 ... Time the player was playing (0 -- 90)
##  $ total_earnings    : int  120000 86000 115000 -3500 158000 43500 56500 195000 9000 35000 ... Points the player earned in a single match (-164000 --- 627000)
##  $ away_shortname    : Factor w/ 18 levels "B04","BMG","BRE",..: 12 18 11 12 2 12 9 12 10 12 ... Away team in a match as 3-character abbreviation (18 teams total)
##  $ away_squad        : int  1562 1579 1572 1562 1566 1562 1570 1562 1573 1562 ... Away squad's 4 digit id (18 teams total)
##  $ away_score        : int  3 1 1 2 0 0 1 3 1 0 ... Goals scored by away team in a given match (0 -- 5)
##  $ time_on_pitch     : int  90 90 90 90 90 90 90 90 90 90 ... same as game_play_duration (0 -- 90)
##  $ status            : Factor w/ 6 levels "bench","not_present",..: 4 4 4 4 4 4 4 4 4 4 ... Player status regarding match participation (bench, not_present, red_card, starter, sub_in, sub_out)
##  $ id                : int  64467 64467 64467 64467 64467 64467 64467 64467 64467 64467 ... Unique player ID (64467 --- 66799)
##  $ match_id          : int  8764 8775 8766 8795 8784 8817 8802 8833 8820 8851 ... Unique match ID (8757 --- 8909)
##  $ period            : Factor w/ 1 level "FullTime": 1 1 1 1 1 1 1 1 1 1 ... Match Status. Since all matches in the dataset are finished, always "Full Time"
##  $ finished          : Factor w/ 1 level "True": 1 1 1 1 1 1 1 1 1 1 ...  Binary match status. Since all matches in the dataset are finished, always "True"
##  $ home_shortname    : Factor w/ 18 levels "B04","BMG","BRE",..: 17 12 12 15 12 4 12 14 12 8 ... Home team as 3-character abbreviation (18 teams total)
##  $ home_squad        : int  1578 1562 1562 1567 1562 1574 1562 1571 1562 1569 ... Home squad's 4 digit id (18 teams total)
##  $ home_score        : int  1 1 2 6 1 2 1 0 0 4 ... Goals scored by home team in a given match (1 -- 6)
##  $ in_for_player     : int  NA NA NA NA NA NA NA NA NA NA ... Player ID of player who was subbed out (NA, see 'id')
##  $ sub_in_timestamp  : int  1439739000 1440250200 1440855000 1442075400 1442669400 1442944800 1443205800 1443965400 1445175000 1445693400 ... Unix Timestamp of substitution
##  $ sub_in_minutes    : int  0 0 0 0 0 0 0 0 0 0 ... Minutes between substitution time and full time. (NA, 0 -- 94)
##  $ sub_out_minutes   : int  NA NA NA NA NA NA NA NA NA NA ... Minutes since start of the match and substitution (NA, 6 -- 94)
##  $ sub_out_timestamp : int  NA NA NA NA NA NA NA NA NA NA ... Unix timestamp of substitution
##  $ shot_on_target    : int  NA NA NA NA NA NA NA NA NA NA ... Count of Shots taken on target by a player (NA, 0 -- 7)
##  $ attempt_saved     : int  6 5 6 5 2 6 4 7 3 9 ... Count of Goal attempts saved by a keeper (NA, 0 -- 10)
##  $ successful_pass   : int  31 31 38 21 20 25 26 14 15 25 ... Count of successful passes by a player (NA, 0 -- 67)
##  $ ID                : int  64467 64467 64467 64467 64467 64467 64467 64467 64467 64467 ... same as 'id'
##  $ Name              : Factor w/ 364 levels "A. Baumjohann",..: 334 334 334 334 334 334 334 334 334 334 ... Player name as a string
##  $ Club              : Factor w/ 18 levels "B04","BMG","BRE",..: 12 12 12 12 12 12 12 12 12 12 ... Club for which the player plays as 3 character abbreviation. NA players have never played a match (NA, 18 clubs total)
##  $ Pos               : Factor w/ 4 levels "ATT","DEF","GOA",..: 3 3 3 3 3 3 3 3 3 3 ... Position which the player plays NA players have never played a match. (NA, GOA, DEF, MID, ATT)
##  $ init_Value        : num  8.5 8.5 8.5 8.5 8.5 8.5 8.5 8.5 8.5 8.5 ... Value of player for which he can be purchased in Millions. NA players have never played a match. (NA, 5 -- 14)
