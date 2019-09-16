function readURL(input) {
  if (input.files && input.files[0]) {

    var reader = new FileReader();

    reader.onload = function(e) {
      // $('.image-upload-wrap').show();
      $('.image-title-wrap').show();
      console.log('Closing img_warp')
//       $('.file-upload-image').attr('src', e.target.result);
      $('.file-upload-content').show();

      $('.image-title').html(input.files[0].name);
      $('.file-upload-btn').show();
    };

    reader.readAsDataURL(input.files[0]);

  } else {
    removeUpload();
  }
}

function removeUpload() {

//   $('.file-upload-btn').hide();
  $('.file-upload-input').replaceWith($('.file-upload-input').clone());
  $('.file-upload-content').show();
  console.log('showing file upload content')
  $('.image-upload-wrap').show();
  $('.imageg-title-wrap').hide();
}
$('.image-upload-wrap').bind('dragover', function () {
		$('.image-upload-wrap').addClass('image-dropping');
	});
	$('.image-upload-wrap').bind('dragleave', function () {
		$('.image-upload-wrap').removeClass('image-dropping');
});

