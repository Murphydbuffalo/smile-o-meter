function changeExpression(mouth, eyes) {
  return function(event) {
    if (mouth.className.includes('frown-mouth')) {
      mouth.className = 'mouth neutral-mouth';
      eyes.forEach((eye) => eye.className = 'eye neutral-eyes');
    } else if (mouth.className.includes('neutral-mouth')) {
      mouth.className = 'mouth smile-mouth';
      eyes.forEach((eye) => eye.className = 'eye smile-eyes');
    } else {
      mouth.className = 'mouth frown-mouth';
      eyes.forEach((eye) => eye.className = 'eye frown-eyes');
    }
  }
}

window.addEventListener('load', function(event) {
  const mouth   = document.querySelector('.mouth');
  const eyes    = document.querySelectorAll('.eye');
  const animate = changeExpression(mouth, eyes);
  const video   = document.querySelector('video');
  const canvas  = document.querySelector('canvas');
  const context = canvas.getContext('2d');

  navigator.mediaDevices
           .getUserMedia({ video: true })
           .then(function(mediaStream) {
             video.srcObject = mediaStream;
             video.play();

             setInterval(function() {
               animate();
               context.drawImage(video, 0, 0, 400, 300);
               const pic = canvas.toDataURL();
               console.log('pic is', pic.slice(0, 50) + '...');
             }, 1000);
           })
           .catch(function(err) {
             console.error('Error getting video stream:', err.message);
           });
});
