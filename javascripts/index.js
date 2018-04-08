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
  document.querySelector('.smiley').onclick = changeExpression(document.querySelector('.mouth'))
  navigator.mediaDevices
           .getUserMedia({ video: true })
           .then(function(mediaStream) {
             const video     = document.querySelector('video');
             video.srcObject = mediaStream;
             video.play();
           })
           .catch(function(err) {
             console.error('Error getting video stream:', err.message);
           });
});
