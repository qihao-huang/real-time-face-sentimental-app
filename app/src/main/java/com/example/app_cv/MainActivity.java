package com.example.app_cv;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button faceDetect = (Button) findViewById(R.id.face_detect);
        TextView login = (TextView)findViewById(R.id.link_to_register);
        faceDetect.setOnClickListener(this);
        login.setOnClickListener(this);
    }

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.link_to_register:
                startActivity(new Intent(this, face_detect.class));
            case R.id.face_detect:
                EditText txt_UserName = (EditText)findViewById(R.id.txt_UserName);
                String username=txt_UserName.getText().toString().trim();
                EditText txt_UserPW = (EditText)findViewById(R.id.txt_UserName);
                String password=txt_UserPW.getText().toString().trim();
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                        != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(this,
                            new String[]{Manifest.permission.CAMERA}, 1);
                }
                else{
                    if(username.isEmpty() || password.isEmpty()){
                        Toast toast1=Toast.makeText(getApplicationContext(),"Please enter your account information!", Toast.LENGTH_SHORT);
                        toast1.show();
                    }
                    else {
                        Intent intent=new Intent(this,face_detect.class);
                        intent.putExtra("data","Hello, "+ username + "!");
                        startActivity(intent);
                        }
                    }
                break;
            default:
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode) {
            case 1:
                if (grantResults.length > 0 &&
                        grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    startActivity(new Intent(this, face_detect.class));
                } else {
                    Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
                }
        }
    }

}
