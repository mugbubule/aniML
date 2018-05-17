<?php

error_reporting(E_ALL);

require_once "vendor/autoload.php";

if (count($argv) < 3) {
  echo "php this.php [first_id] [range]\n";
  exit(1);
}

$anime_fd = fopen("anime.csv", "a");
$voice_actor_fd = fopen("voice_actor.csv", "a");
$producer_fd = fopen("producer.csv", "a");
$licensor_fd = fopen("licensor.csv", "a");
$studio_fd = fopen("studio.csv", "a");
$staff_fd = fopen("staff.csv", "a");
echo "witness as I painfully try to scrap MAL\n";

$id = (int)$argv[1];
$first_id = $id;
$range = $argv[1] + $argv[2] * 2;

while ($id <= $first_id + $range) {
  echo "Trying to get " . $id;
  try {
      scrapAnime($id, $anime_fd, $voice_actor_fd, $producer_fd, $licensor_fd, $studio_fd, $staff_fd);
      $id += 2;
      echo " done! Waiting for next request";
    } catch (Exception $e) {
      echo ' Caught exception: ', $e->getMessage;
      echo ' ';
      echo $id;
      if ($e->getMessage != "File does not exist") {
        close_everything($anime_fd, $voice_actor_fd, $producer_fd, $licensor_fd, $studio_fd, $staff_fd);
        exit(0);
      }
    }
    echo "\n";
  sleep(5);
}
echo "I'm done\n"

fclose($staff_fd);

function close_everything($anime_fd, $voice_actor_fd, $producer_fd, $licensor_fd, $studio_fd, $staff_fd) {
  fclose($anime_fd);
  fclose($voice_actor_fd);
  fclose($staff_fd);
  fclose($producer_fd);
  fclose($licensor_fd);
  fclose($studio_fd);
}

function scrapAnime($id, $anime_fd, $voice_actor_fd, $producer_fd, $licensor_fd, $studio_fd, $staff_fd) {

  $anime = new Jikan\Jikan;
  $anime->Anime($id, [STATS, CHARACTERS_STAFF]);
  $line = "";

  $line .= $anime->response["mal_id"] . ", ";
  $line .= $anime->response["title"] . ", ";
  $line .= $anime->response["type"] . ", ";
  $line .= $anime->response["source"] . ", ";
  $line .= $anime->response["episodes"] . ", ";
  $line .= $anime->response["aired"]["from"] . ", "; //the older the less important
  $line .= $anime->response["duration"] . ", ";
  $line .= $anime->response["rating"] . ", ";
  $line .= $anime->response["score"] . ", ";
  $line .= $anime->response["rank"] . ", ";
  $line .= $anime->response["scored_by"] . ", ";
  $line .= $anime->response["popularity"] . ", ";
  $line .= $anime->response["members"] . ", ";
  $line .= $anime->response["favorites"] . ", ";

  $related_work = 0;
  foreach ($anime->response["related"] as $related) {
    $related_work++;
    $related_work += count($related);
  }
  $line .= $related_work . ", ";

  foreach ($anime->response["producer"] as $producer) {
    fwrite($producer_fd, $producer["name"] . ", " . $anime->response["mal_id"] . "\n");
  }
  foreach ($anime->response["licensor"] as $licensor) {
    fwrite($licensor_fd, $licensor["name"] . ", " . $anime->response["mal_id"] . "\n");
  }
  foreach ($anime->response["studio"] as $studio) {
    fwrite($studio_fd, $studio["name"] . ", " . $anime->response["mal_id"] . "\n");
  }

  $line .= "\"";
  foreach ($anime->response["genre"] as $genre) {
    $line .= $genre["name"] . ", ";
  }
  $line .= "\", ";

  $line .= $anime->response["watching"] . ", ";
  $line .= $anime->response["completed"] . ", ";
  $line .= $anime->response["on_hold"] . ", ";
  $line .= $anime->response["dropped"] . ", ";
  $line .= $anime->response["plan_to_watch"] . ", ";
  $line .= $anime->response["total"]; //most important actually
  //$line .= $anime->response["stats"] . ", "; //if usefull (basically how people voted from 0 to 10)

  foreach ($anime->response["character"] as $character) { //undefined etc
    foreach($character["voice_actor"] as $voice_actor) {
      if ($voice_actor["language"] == "Japanese") {
        fwrite($voice_actor_fd, $voice_actor["mal_id"] . ", " . $voice_actor["name"] . ", " . $anime->response["mal_id"] . "\n");
      }
    }
  }

  foreach ($anime->response["staff"] as $staff) {
    fwrite($staff_fd, $staff["mal_id"] . ", " . $staff["name"] . ", \"" . $staff["role"] . "\", " . $anime->response["mal_id"] . "\n");
  }
    fwrite($anime_fd, $line . "\n");
}
?>
