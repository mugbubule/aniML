<?php
require_once dirname(__DIR__) . "/vendor/autoload.php";

if ($argv.count() < 3) {
  echo "this.php [first_id] [range]";
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

while ($id < $first_id + $range) {
  echo "Trying to get " . $id;
  try {
      scrapAnime($id, $anime_fd, $voice_actor_fd, $producer_fd, $licensor_fd, $studio_fd, $staff_fd);
    } catch (Exception $e) {
      echo 'Caught exception: ',  $e->getMessage(); // "File does not exist" (the anime with this ID doesn't exist on MAL)
      echo $id;
      close_everything($anime_fd, $voice_actor_fd, $producer_fd, $licensor_fd, $studio_fd, $staff_fd);
      exit(0);
    }
  sleep(5);
  $id += 2;
}

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

  $line .= $anime["mal_id"] . ", ";
  $line .= $anime["title"] . ",";
  $line .= $anime["type"] . ",";
  $line .= $anime["source"] . ",";
  $line .= $anime["episodes"] . ",";
  $line .= $anime["aired"]["from"] . ","; //the older the less important
  $line .= $anime["duration"] . ",";
  $line .= $anime["rating"] . ",";
  $line .= $anime["score"] . ",";
  $line .= $anime["rank"] . ",";
  $line .= $anime["scored_by"] . ",";
  $line .= $anime["popularity"] . ",";
  $line .= $anime["members"] . ",";
  $line .= $anime["favorites"] . ",";

  $related_work = 0;
  foreach ($anime["related"] as $related) {
    $related_work++;
    foreach ($related as $work) {
      $related_work++;
    }
  }
  $line .= $related_work . ",";

  foreach ($anime["producer"] as $producer) {
    fwrite($producer_fd, $producer["name"] . ", " . $anime["mal_id"] . "\n");
  }
  foreach ($anime["licensor"] as $licensor) {
    fwrite($licensor_fd, $licensor["name"] . ", " . $anime["mal_id"] . "\n");
  }
  foreach ($anime["studio"] as $studio) {
    fwrite($studio_fd, $studio["name"] . ", " . $anime["mal_id"] . "\n");
  }

  $line .= "\"";
  foreach ($anime["genre"] as $genre) {
    $line .= $genre["name"] . ",";
  }
  $line .= "\"";

  $line .= $anime["genre"] . ",";
  $line .= $anime["watching"] . ",";
  $line .= $anime["completed"] . ",";
  $line .= $anime["on_hold"] . ",";
  $line .= $anime["dropped"] . ",";
  $line .= $anime["plan_to_watch"] . ",";
  $line .= $anime["total"] . ","; //most important actually
  $line .= $anime["stats"] . ",";

  foreach ($anime["characters"] as $character) {
    foreach($character["voice_actor"] as $voice_actor) {
      if ($voice_actor["language"] == "Japanese") {
        fwrite($voice_actor_fd, $voice_actor["mal_id"] . ", " . $voice_actor["name"] . ", " . $anime["mal_id"] . "\n")
      }
    }
  }

  foreach ($anime["staff"] as $staff) {
    fwrite($staff_fd, $staff["mal_id"] . ", " . $staff["name"] . ", " . $staff["role"] . ", " . $anime["mal_id"] . "\n");
  }
    fwrite($anime_fd, $line . "\n");
}
?>

