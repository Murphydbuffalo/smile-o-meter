function changeExpression(mouth) {
  return function(event) {
    if (mouth.className.includes('frown')) {
      mouth.className = 'mouth neutral';
    } else if (mouth.className.includes('neutral')) {
      mouth.className = 'mouth smile';
    } else {
      mouth.className = 'mouth frown';
    }
  }
}

window.addEventListener("load", function(event) {
  const mouth = document.querySelector('.mouth');
  document.querySelector('.smiley').onclick = changeExpression(mouth)
});
